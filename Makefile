.PHONY: demo demo-mppi test

demo:
	python3 -B -m planner_lab.run --out outputs/basic

demo-mppi:
	python3 -B -m planner_lab.run_mppi --out outputs/mppi --scenarios crossing,corridor --seeds 1 --max-steps 130 --rollouts 44 --horizon 16

test:
	python3 -B -m unittest discover -s tests -v
