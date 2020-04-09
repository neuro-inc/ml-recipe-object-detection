include Makefile

CMD_PREPARE=\
  export DEBIAN_FRONTEND=noninteractive && \
  apt-get -qq update && \
  apt-get -qq install -y --no-install-recommends pandoc >/dev/null

CMD_NBCONVERT=\
  $(JUPYTER_SCREEN) \
  jupyter nbconvert \
  --execute \
  --no-prompt \
  --no-input \
  --to=asciidoc \
  --ExecutePreprocessor.timeout=600 \
  --output=/tmp/out \
  $(PROJECT_PATH_ENV)/$(NOTEBOOKS_DIR)/demo.ipynb && \
  echo "Test succeeded: PROJECT_PATH_ENV=$(PROJECT_PATH_ENV) TRAINING_MACHINE_TYPE=$(TRAINING_MACHINE_TYPE)"


.PHONY: _generate_random_label
_generate_random_label:
	@cat /dev/urandom | tr -dc 'a-z0-9' | head -n1 -c16

.PHONY: test_jupyter
test_jupyter: JUPYTER_CMD=bash -c '$(CMD_PREPARE) && $(CMD_NBCONVERT)'
# recipe's jupyter doesn't work on CPU
test_jupyter: TRAINING_MACHINE_TYPE=gpu-small
test_jupyter: jupyter
	# kill job to set its SUCCEEDED status in platform-api
	make kill-jupyter

.PHONY: test_jupyter_baked
test_jupyter_baked: PROJECT_PATH_ENV=/project-local
test_jupyter_baked: JOB_NAME=jupyter-baked-$(PROJECT_POSTFIX)
test_jupyter_baked:
    # recipe's jupyter doesn't work on CPU
	$(NEURO) run $(RUN_EXTRA) \
		--name $(JOB_NAME) \
		--preset gpu-small \
		$(CUSTOM_ENV_NAME) \
		bash -c '$(CMD_PREPARE) && $(CMD_NBCONVERT)'
	# kill job to set its SUCCEEDED status in platform-api
	$(NEURO) kill $(JOB_NAME) || :