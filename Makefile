watch:
	./utils/watch

submit:
	qsub job.sh

cancel:
	./utils/cancel

.PHONEY: watch submit cancel
