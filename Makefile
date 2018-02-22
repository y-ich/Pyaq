FREEZE_GRAPH = ~/OpenSources/tensorflow/tensorflow/python/tools/freeze_graph.py
TARGET = pre_train/frozen_model.pb
GRAPH = pre_train/graph.pb
CHECK_POINT = pre_train/model.ckpt

all: $(TARGET)

$(GRAPH): save_graph.py model.py
	python2 save_graph.py

$(TARGET): $(GRAPH) $(CHECK_POINT)
	python2 $(FREEZE_GRAPH) --input_graph=$(GRAPH) --input_checkpoint=$(CHECK_POINT) --output_graph=$(TARGET) --output_node_names=pfc/policy,vfc/value
