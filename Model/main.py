"""Main interactive menu for VeriFact tooling."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
RETRIEVER_SCRIPT = ROOT_DIR / "retrieval" / "run_retriever.py"
DOCKERFILE_RETRIEVER = ROOT_DIR / "Dockerfile.retriever"
IMAGE_TAG = "verifact-retriever:latest"


def run_command(command: list[str]) -> None:
	subprocess.run(command, cwd=str(ROOT_DIR), check=True)


def run_retriever_interactive() -> None:
	if not RETRIEVER_SCRIPT.exists():
		print("Retriever script not found.")
		return
	run_command([sys.executable, str(RETRIEVER_SCRIPT)])


def docker_up() -> None:
	run_command(["docker", "compose", "up", "-d", "--build"])


def docker_reindex() -> None:
	run_command([
		"docker",
		"compose",
		"run",
		"--rm",
		"--workdir",
		"/app/retrieval",
		"retriever",
		"python",
		"run_retriever.py",
		"setup",
	])


def docker_build_retriever() -> None:
	if not DOCKERFILE_RETRIEVER.exists():
		print("Dockerfile.retriever not found.")
		return
	run_command([
		"docker",
		"build",
		"-t",
		IMAGE_TAG,
		"-f",
		str(DOCKERFILE_RETRIEVER),
		".",
	])


def docker_menu() -> None:
	while True:
		print("\nDocker Services")
		print("a. Initialize and run ElasticSearch (and other images)")
		print("b. Create retriever Docker image (and update)")
		print("c. Reindex passages in ElasticSearch")
		print("d. Back")

		choice = input("\nSelect an option: ").strip().lower()
		if choice == "a":
			docker_up()
			print("Docker services started.")
		elif choice == "b":
			docker_build_retriever()
			print(f"Built image: {IMAGE_TAG}")
		elif choice == "c":
			docker_reindex()
			print("Reindex complete.")
		elif choice == "d":
			break
		else:
			print("Invalid choice.")


def main() -> None:
	while True:
		print("\nMain Menu")
		print("1. Run Retriever system")
		print("2. Docker Services")
		print("3. Exit")

		choice = input("\nSelect an option: ").strip().lower()
		if choice == "1":
			run_retriever_interactive()
		elif choice == "2":
			docker_menu()
		elif choice == "3":
			print("Goodbye.")
			break
		else:
			print("Invalid choice.")


if __name__ == "__main__":
	main()
