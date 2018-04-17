#pragma once
#ifndef LOGGER_H
#define LOGGER_H

class logger {
public:
	bool print_to_file;
	bool verbose;
	logger(){print_to_file=false;verbose=true;}	// Opens a file for use with ofstream fout.
	~logger(){fout.close();}			// Flushes and closes the logging file.
	void set_ptf(bool t) {print_to_file=t;};
	void set_v(bool v) {verbose=v;};
	void open(std::string filename){
			fout.open(filename);
			/*if(fout.is_open()) {
				std::cout << "opened log file successfully\n";
				return 1;
			}
			else {
				std::cout << "Failed to open log file.\n";
				return 0;
			}*/
	}
	bool is_open(){return fout.is_open();}
	void close(){fout.close();}
	//logger& operator(); // Allows us to set the mask in the same line as the messages

	template<class T>
	friend logger& operator<<( logger& log, const T& output );


	std::ofstream fout;
	//unsigned mask;
};

template<class T>
logger& operator<<( logger& log, const T& output ) {

	if(log.verbose)
		std::cout << output;
	if (log.print_to_file)
		log.fout << output;

	return log;
}
#endif
