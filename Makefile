watch:
	./bdrl/utils/watch

ssubmit:
	sbatch slurmjob.sh

submit:
	qsub job.sh

cancel:
	./bdrl/utils/cancel

int:
	srun --nodes=1 --ntasks-per-node=1 --time=01:00:00 --partition=veryshort --pty bash -i


.PHONEY: watch submit cancel
