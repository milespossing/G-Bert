pull:
	docker pull mpossing/dlh-final:latest

runBertTune:
	docker run -p 6006:6006 -v $(shell pwd):/opt/project -w /opt/project/code --rm --gpus all \
		mpossing/dlh-final:latest python run_bert_tune.py


runFromConfig:
	docker run -p 6006:6006 -v $(shell pwd):/opt/project -w /opt/project/code --rm --gpus all \
		mpossing/dlh-final:latest python run_from_config.py

runLrTune:
	docker run -p 6006:6006 -v $(shell pwd):/opt/project -w /opt/project/code --rm --gpus all \
		mpossing/dlh-final:latest python run_lr_tune.py
