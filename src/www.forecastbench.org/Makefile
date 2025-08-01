all :
	$(MAKE) clean
	$(MAKE) deploy

.PHONY : all clean deploy

UPLOAD_DIR = upload

.gcloudignore:
	cp -r $(ROOT_DIR)src/helpers/.gcloudignore .

entrypoint.sh: entrypoint.sh.template
	sed \
		-e 's/WEBSITE_BUCKET/$(WEBSITE_BUCKET)/g' \
		$< > entrypoint.sh

deploy : entrypoint.sh Dockerfile .gcloudignore
	mkdir -p $(UPLOAD_DIR)
	rsync -a \
		--exclude='Makefile' \
		--exclude='entrypoint.sh.template' \
		--exclude='_site/' \
		--exclude='upload/' \
		--exclude='assets/js/leaderboard_compact.js' \
		--exclude='assets/js/leaderboard_full.js' \
		./ $(UPLOAD_DIR)/
	gcloud run jobs deploy \
		func-website \
		--project $(CLOUD_PROJECT) \
		--region $(CLOUD_DEPLOY_REGION) \
		--tasks 1 \
		--task-timeout 2m \
		--memory 4Gi \
		--max-retries 0 \
		--cpu 1 \
		--service-account $(QUESTION_BANK_BUCKET_SERVICE_ACCOUNT) \
		--set-env-vars $(DEFAULT_CLOUD_FUNCTION_ENV_VARS),BUCKET_TO_COMPRESS= \
		--source $(UPLOAD_DIR)

clean :
	rm -rf $(UPLOAD_DIR) .gcloudignore entrypoint.sh
