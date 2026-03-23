CXX      := g++
TARGET   := fast-wc
SRC      := src/fast-wc.cpp

CXXFLAGS := -O3 -march=native -static -s -fno-rtti \
            -ffunction-sections -fdata-sections \
            -ffast-math -std=c++23

LDFLAGS  := -Wl,--gc-sections

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
