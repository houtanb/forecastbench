all :
	$(MAKE) clean
	$(MAKE) deploy

.PHONY : all clean deploy

UPLOAD_DIR = upload

.gcloudignore:
	cp -r $(ROOT_DIR)src/helpers/.gcloudignore .

deploy : entrypoint.sh Dockerfile .gcloudignore
	mkdir -p $(UPLOAD_DIR)
	cp -r $(ROOT_DIR)utils $(UPLOAD_DIR)/
	cp -r $(ROOT_DIR)src/helpers $(UPLOAD_DIR)/
	rsync -a --exclude='Makefile' --exclude='_site/' --exclude='upload/' --exclude='assets/js/leaderboard_compact.js' --exclude='assets/js/leaderboard_full.js' ./ $(UPLOAD_DIR)/
	gcloud run jobs deploy \
		func-website \
		--project $(CLOUD_PROJECT) \
		--region $(CLOUD_DEPLOY_REGION) \
		--tasks 1 \
		--task-timeout 1m \
		--memory 4Gi \
		--max-retries 0 \
		--cpu 1 \
		--add-volume name=$(WEBSITE_BUCKET),type=cloud-storage,bucket=$(WEBSITE_BUCKET),readonly=false \
		--add-volume-mount volume=$(WEBSITE_BUCKET),mount-path=$(BUCKET_MOUNT_POINT)/$(WEBSITE_BUCKET) \
		--service-account $(QUESTION_BANK_BUCKET_SERVICE_ACCOUNT) \
		--set-env-vars $(DEFAULT_CLOUD_FUNCTION_ENV_VARS),BUCKET_TO_COMPRESS= \
		--source $(UPLOAD_DIR)

clean :
	rm -rf $(UPLOAD_DIR) .gcloudignore
