all:
	$(MAKE) clean
	$(MAKE) deploy

.PHONY: all clean deploy

UPLOAD_DIR = upload

.gcloudignore:
	cp -r $(ROOT_DIR)src/helpers/.gcloudignore .

Dockerfile: $(ROOT_DIR)src/helpers/Dockerfile.template
	sed \
		-e 's/REGION/$(CLOUD_DEPLOY_REGION)/g' \
		-e 's/STACK/google-22-full/g' \
		-e 's/PYTHON_VERSION/python312/g' \
		$< > Dockerfile

deploy : main.py .gcloudignore requirements.txt Dockerfile
	rm -rf $(UPLOAD_DIR)
	mkdir -p $(UPLOAD_DIR)
	cat $(ROOT_DIR)requirements.runtime.txt requirements.txt > $(UPLOAD_DIR)/requirements.txt
	cp -r $(ROOT_DIR)src/helpers $(UPLOAD_DIR)/
	cp -r $(ROOT_DIR)src/sources $(UPLOAD_DIR)/
	cp -r $(ROOT_DIR)src/llm_forecaster $(UPLOAD_DIR)/
	mkdir -p $(UPLOAD_DIR)/orchestration
	cp $(ROOT_DIR)src/orchestration/__init__.py $(UPLOAD_DIR)/orchestration/
	cp $(ROOT_DIR)src/orchestration/_io.py $(UPLOAD_DIR)/orchestration/
	cp $(ROOT_DIR)src/_fb_types.py $(UPLOAD_DIR)/
	cp $(ROOT_DIR)src/_schemas.py $(UPLOAD_DIR)/
	cp main.py $(UPLOAD_DIR)/main.py
	cp Dockerfile $(UPLOAD_DIR)/
	gcloud run jobs deploy \
		func-llm-forecaster-manager \
		--project $(CLOUD_PROJECT) \
		--region $(CLOUD_DEPLOY_REGION) \
		--tasks 1 \
		--parallelism 1 \
		--task-timeout 24h \
		--memory 2Gi \
		--max-retries 0 \
		--service-account $(WORKFLOW_SERVICE_ACCOUNT) \
		--set-env-vars $(DEFAULT_CLOUD_FUNCTION_ENV_VARS),TEST_OR_PROD=$(if $(filter $(BUILD_ENV),prod),PROD,TEST) \
		--source $(UPLOAD_DIR)

clean:
	rm -rf $(UPLOAD_DIR) .gcloudignore Dockerfile
