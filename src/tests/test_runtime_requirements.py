import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_REQUIREMENT = "git+https://github.com/forecastingresearch/utils@"
LEGACY_BASELINE_ROOT = "/".join(("src", "base_eval", "llm_baselines"))
SHARED_LLM_PROVIDER_REQUIREMENT_NAMES = {
    "anthropic",
    "google-genai",
    "openai",
    "together",
}


def iter_deploy_requirements():
    for requirements_path in sorted(ROOT.glob("src/**/requirements.txt")):
        if any(part == "upload" or part.startswith("upload-") for part in requirements_path.parts):
            continue
        makefile_path = requirements_path.with_name("Makefile")
        if makefile_path.exists():
            yield requirements_path, makefile_path


def test_shared_utils_pin_only_lives_in_root_runtime_requirements():
    root_runtime_requirements = (ROOT / "requirements.runtime.txt").read_text()

    assert RUNTIME_REQUIREMENT in root_runtime_requirements
    for requirements_path, _makefile_path in iter_deploy_requirements():
        assert RUNTIME_REQUIREMENT not in requirements_path.read_text(), requirements_path


STAGED_RUNTIME_REQUIREMENT = re.compile(
    r"cat \$\(ROOT_DIR\)requirements\.runtime\.txt requirements\.txt"
    r" > (?P<upload_dir>\$1|\$\([^)]+\))/requirements\.txt"
)


def test_deploy_makefiles_stage_shared_runtime_requirements():
    for _requirements_path, makefile_path in iter_deploy_requirements():
        makefile = makefile_path.read_text()

        assert STAGED_RUNTIME_REQUIREMENT.search(makefile), makefile_path


def test_deploy_makefiles_clear_upload_dir_before_staging_requirements():
    for _requirements_path, makefile_path in iter_deploy_requirements():
        makefile = makefile_path.read_text()
        for match in STAGED_RUNTIME_REQUIREMENT.finditer(makefile):
            upload_dir = match.group("upload_dir")
            assert f"rm -rf {upload_dir}" in makefile[: match.start()], makefile_path


def test_deploy_requirements_do_not_duplicate_utils_owned_llm_provider_deps():
    for requirements_path, _makefile_path in iter_deploy_requirements():
        requirements = requirements_path.read_text().splitlines()
        requirement_names = {
            re.split(r"[<>=!~]", requirement, maxsplit=1)[0].strip() for requirement in requirements
        }
        assert requirement_names.isdisjoint(
            SHARED_LLM_PROVIDER_REQUIREMENT_NAMES
        ), requirements_path


def test_metadata_deploy_requirements_keep_direct_gcp_deps():
    deploy_dirs = [
        ROOT / "src/metadata/tag_questions",
        ROOT / "src/metadata/validate_questions",
    ]

    for deploy_dir in deploy_dirs:
        requirements = (deploy_dir / "requirements.txt").read_text().splitlines()

        assert "google-cloud-storage" in requirements
        assert "google-cloud-secret-manager" in requirements


def test_root_makefile_routes_llm_baseline_targets_to_refactored_jobs():
    makefile = (ROOT / "Makefile").read_text()

    assert "$(MAKE) -C src/orchestration/func_llm_forecaster_manager" in makefile
    assert "$(MAKE) -C src/orchestration/func_llm_forecaster_worker" in makefile
    assert f"{LEGACY_BASELINE_ROOT}/manager" not in makefile
    assert f"{LEGACY_BASELINE_ROOT}/worker" not in makefile


def test_root_make_test_does_not_install_project_requirements():
    result = subprocess.run(
        ["make", "--dry-run", "test", "ARGS=--version"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    dry_run_output = result.stdout + result.stderr

    assert "python -m pytest src/tests/ --version" in dry_run_output
    assert "pip install" not in dry_run_output
    assert ".root-requirements-installed" not in dry_run_output
    assert ".all-requirements-installed" not in dry_run_output


def test_root_install_requirements_uses_deployed_utils_pin():
    makefile = (ROOT / "Makefile").read_text()

    assert "ROOT_REQUIREMENTS_NO_RUNTIME := .venv/.root-requirements-no-runtime.txt" in makefile
    assert "LOCAL_UTILS_PATH" not in makefile
    assert (
        "$(ROOT_REQUIREMENTS_STAMP): requirements.txt requirements.runtime.txt | $(VENV_PYTHON)"
        not in makefile
    )
    assert (
        "$(ROOT_REQUIREMENTS_STAMP): Makefile requirements.txt requirements.runtime.txt |"
        in makefile
    )
    assert (
        "$(ALL_REQUIREMENTS_STAMP): Makefile $(ROOT_REQUIREMENTS_STAMP) $(SRC_REQUIREMENTS)"
        in makefile
    )
    assert "python -m pip install -e" not in makefile
    assert "python -m pip install -r requirements.runtime.txt" in makefile
    assert "python -m pip install -r $(ROOT_REQUIREMENTS_NO_RUNTIME)" in makefile
