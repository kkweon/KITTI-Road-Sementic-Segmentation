.PHONY: clean train board

SUMMARY_DIR=logs
EPOCH=50

train: clean
	python main.py --logdir ${SUMMARY_DIR} --epoch ${EPOCH}

board:
	tensorboard --logdir ${SUMMARY_DIR}

clean:
	rm -rf runs
	rm -rf ${SUMMARY_DIR}
	rm -rf builder
