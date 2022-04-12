import numpy as np
import torch
import torch.nn as nn

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

class ModelWrapper(nn.Module):
    """""" 
    def __init__(self,
                env=None,
                embedder=None,
                encoder=None,
                decoder=None,
                beam_type="search",
                beam_length_penalty=1,
                beam_size=1,
                beam_early_stopping=True,
                max_generated_output_len=200,
                beam_temperature=1.,
                ):
        super().__init__()

        self.env = env
        self.embedder = embedder
        self.encoder = encoder
        self.decoder = decoder
        self.beam_type = beam_type
        self.beam_early_stopping = beam_early_stopping
        self.max_generated_output_len = max_generated_output_len
        self.beam_size = beam_size
        self.beam_length_penalty = beam_length_penalty
        self.beam_temperature = beam_temperature
        self.device=next(self.embedder.parameters()).device

    @torch.no_grad()
    def forward(
        self,
        input,
    ):

        """
        x: bags of sequences (B, T)
        """

        env = self.env
        embedder, encoder, decoder = self.embedder, self.encoder, self.decoder

        B, T = len(input), max([len(xi) for xi in input])
        outputs = []
        for chunk in chunks(np.arange(B), min(int(10000/T), int(100000/self.beam_size/self.max_generated_output_len))):
            x, x_len = embedder([input[idx] for idx in chunk])
            encoded = encoder("fwd", x=x, lengths=x_len, causal=False).transpose(0,1)
            bs = encoded.shape[0]

            ### Greedy solution.
            generations, _ = decoder.generate(
                        encoded,
                        x_len,
                        sample_temperature=None,
                        max_len=self.max_generated_output_len,
                        )  

            generations = generations.unsqueeze(-1).view(generations.shape[0], bs, 1)
            generations = generations.transpose(0,1).transpose(1,2).cpu().tolist()
            generations = [list(filter(lambda x: x is not None, [env.idx_to_infix(hyp[1:-1], is_float=False, str_array=False) for hyp in generations[i]])) for i in  range(bs)]

            if self.beam_type == "search":
                _, _, search_generations = decoder.generate_beam(
                    encoded,
                    x_len,
                    beam_size=self.beam_size,
                    length_penalty=self.beam_length_penalty,
                    max_len=self.max_generated_output_len,
                    early_stopping=self.beam_early_stopping,
                ) 
                search_generations = [sorted([hyp for hyp in search_generations[i].hyp], key=lambda s: s[0], reverse=True) for i in range(bs)]
                search_generations = [list(filter(lambda x: x is not None, [env.idx_to_infix(hyp.cpu().tolist()[1:], is_float=False, str_array=False) for (_, hyp) in search_generations[i]])) for i in range(bs)]
                for i in range(bs):
                    generations[i].extend(search_generations[i])

            elif self.beam_type == "sampling":
                num_samples = self.beam_size
                encoded = (encoded.unsqueeze(1)
                    .expand((bs, num_samples) + encoded.shape[1:])
                    .contiguous()
                    .view((bs * num_samples,) + encoded.shape[1:])
                )
                x_len = x_len.unsqueeze(1).expand(bs, num_samples).contiguous().view(-1)
                sampling_generations, _ = decoder.generate(
                    encoded,
                    x_len,
                    sample_temperature = self.beam_temperature,
                    max_len=self.max_generated_output_len
                    )  
                sampling_generations = sampling_generations.unsqueeze(-1).view(sampling_generations.shape[0], bs, num_samples)
                sampling_generations = sampling_generations.transpose(0, 1).transpose(1, 2).cpu().tolist()
                sampling_generations = [list(filter(lambda x: x is not None, [env.idx_to_infix(hyp[1:-1], is_float=False, str_array=False) for hyp in sampling_generations[i]])) for i in range(bs)]
                for i in range(bs):
                    generations[i].extend(sampling_generations[i])
            else: 
                raise NotImplementedError
            outputs.extend(generations)
        return outputs