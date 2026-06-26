CXX = clang++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Wpedantic -pthread
SRCDIR = src
OBJDIR = obj
TARGET = secom_engine

SRCS = $(SRCDIR)/main.cpp \
       $(SRCDIR)/preprocessor.cpp \
       $(SRCDIR)/domain_enricher.cpp \
       $(SRCDIR)/models.cpp \
       $(SRCDIR)/threshold_tuner.cpp \
       $(SRCDIR)/quantizer.cpp \
       $(SRCDIR)/engine.cpp

OBJS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRCS))

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -c -o $@ $<

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR) $(TARGET)
