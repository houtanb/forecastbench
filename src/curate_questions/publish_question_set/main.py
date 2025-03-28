"""Publish question set in git repo."""

import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from helpers import dates, decorator, env, git, keys, question_curation  # noqa: E402

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from utils import gcp  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@decorator.log_runtime
def driver(_):
    """Publish question sets to git.

    If today is the forecast due date, publish the question set to git.
    """
    if not question_curation.is_today_question_set_publication_date():
        logger.info("Today is NOT the question set publication date.")
        return

    remote_folder = "datasets/question_sets"
    local_folder = f"/tmp/{remote_folder}"
    forecast_due_date = dates.get_date_today_as_iso()
    question_sets = gcp.storage.list(bucket_name=env.QUESTION_SETS_BUCKET)
    question_set = f"{forecast_due_date}-llm.json"

    if question_set not in question_sets:
        raise FileNotFoundError(
            f"Question set {question_set} not found in bucket {env.QUESTION_SETS_BUCKET}."
        )

    logger.info(f"Found {question_set}. Downloading and pushing to git.")
    local_filename = f"{local_folder}/{question_set}"
    os.makedirs(local_folder, exist_ok=True)
    gcp.storage.download(
        bucket_name=env.QUESTION_SETS_BUCKET,
        filename=question_set,
        local_filename=local_filename,
    )

    # Create latest-llm.json soft link
    soft_link_question_set = "latest-llm.json"
    soft_link_filename = f"{local_folder}/{soft_link_question_set}"
    if os.path.exists(soft_link_filename):
        os.remove(soft_link_filename)
    os.symlink(question_set, soft_link_filename)

    mirrors = keys.get_secret_that_may_not_exist("HUGGING_FACE_REPO_URL")
    mirrors = [mirrors] if mirrors else []
    git.clone_and_push_files(
        repo_url=keys.API_GITHUB_DATASET_REPO_URL,
        files={
            local_filename: f"{remote_folder}/{question_set}",
            soft_link_filename: f"{remote_folder}/{soft_link_question_set}",
        },
        commit_message=f"publish {question_set}.",
        mirrors=mirrors,
    )

    logger.info("Done.")


if __name__ == "__main__":
    driver(None)
