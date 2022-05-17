AR:=ar
CC:=clang
SRCDIR:=src
OBJDIR:=build
BINDIR:=dist

INCS:=$(wildcard $(SRCDIR)/*.h)

MAKEFLAGS:=-j
CFLAGS:=-std=gnu11 -Wall -Wextra -pedantic -march=native -O3 -fopenmp -g
IFLAGS:=
LFLAGS:=-lm

ALG_OBJS:=$(OBJDIR)/gp.o $(OBJDIR)/rng.o
MAIN_OBJS:=$(OBJDIR)/main.o $(OBJDIR)/cmd_args.o $(OBJDIR)/readline.o

BIN:=$(BINDIR)/regressor

all: $(BIN)

$(BIN): $(MAIN_OBJS) $(ALG_OBJS)
	@echo linking $@ from $^
	@mkdir -p $(BINDIR)
	@$(CC) $(CFLAGS) $^ -o $@ $(LFLAGS)

$(OBJDIR)/%.o : $(SRCDIR)/%.c $(INCS)
	@echo compiling $< into $@
	@mkdir -p $(OBJDIR)
	@$(CC) $(CFLAGS) $(IFLAGS) -c $< -o $@

clean:
	@rm -rf $(OBJDIR)

nuke: clean
	@rm -rf $(INCDIR) $(BINDIR)

strip: all
	@echo running strip on $(BIN)
	@strip $(BIN)
