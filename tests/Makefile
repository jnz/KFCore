# Makefile to generate the test binary
# KFCore, Jan Zwiener (jan@zwiener.org)

CC = gcc
CFLAGS = -O2 -Wall -Wdouble-promotion -fno-math-errno
CFLAGS += -I/usr/include/eigen3
CFLAGS += -I../c/ -I../cpp -Ibenchmark -I../c/navigation_tools

SOURCES += test.c
SOURCES += ../c/navigation_tools/navtoolbox.c
SOURCES += ../c/linalg.c
SOURCES += ../c/miniblas.c
SOURCES += ../c/kalman_udu.c
SOURCES += ../c/kalman_takasu.c
SOURCES += ../cpp/kalman_takasu.cpp
SOURCES += benchmark/benchmark.cpp

HEADERS = linalg.h
HEADERS += miniblas.h
HEADERS += kalman_takasu.h
HEADERS += kalman_udu.h
HEADERS += navtoolbox.h
HEADERS += benchmark/benchmark.h

OUTPUT = kfcoretest
LDFLAGS = -lm -lstdc++

default: run_test

all: $(OUTPUT)

$(OUTPUT): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES)  -o $(OUTPUT) $(LDFLAGS)

run_test: $(OUTPUT)
	./$(OUTPUT)
	@RETVAL=$$?; \
	if [ $$RETVAL -ne 0 ]; then \
		echo "Test failed with return value $$RETVAL"; \
		exit 1; \
	else \
		echo "Test passed"; \
	fi

asm: $(SOURCES)
	$(CC) $(CFLAGS) -fverbose-asm -S $(SOURCES)

.PHONY: profile prof analyze tidy format cppcheck clean
analyze:
	scan-build -v clang -fsanitize=address $(CFLAGS) $(SOURCES) -o $(OUTPUT) $(LDFLAGS)

profile: prof
	./$(OUTPUT)
	gprof $(OUTPUT) gmon.out > analysis.txt

prof: CFLAGS += -pg
prof: LDFLAGS += -pg
prof:
	$(CC) $(CFLAGS) $(SOURCES) -o $(OUTPUT) $(LDFLAGS)

tidy:
	clang-tidy $(SOURCES) --

format: $(SOURCES) $(HEADERS)
	clang-format -i $(SOURCES)
	clang-format -i $(HEADERS)

cppcheck:
	cppcheck --enable=all --inconclusive --error-exitcode=1 $(SOURCES)

clean:
	rm -f $(OUTPUT) *.s gmon.out analysis.txt
