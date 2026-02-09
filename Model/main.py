"""Main interactive menu for VeriFact tooling."""

from __future__ import annotations

import subprocess
import socket
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
RETRIEVER_SCRIPT = ROOT_DIR / "retrieval" / "run_retriever.py"
LABELING_SCRIPT = ROOT_DIR / "labelling" / "label_passages.py"
DOCKERFILE_RETRIEVER = ROOT_DIR / "Dockerfile.retriever"
IMAGE_TAG = "verifact-retriever:latest"


def run_command(command: list[str], capture: bool = True) -> bool:
	result = subprocess.run(
		command,
		cwd=str(ROOT_DIR),
		capture_output=capture,
		text=True
	)

	if result.returncode != 0:
		print("Command failed.")
		if capture:
			if result.stdout:
				print(result.stdout.strip())
			if result.stderr:
				print(result.stderr.strip())
		return False
	return True


def run_retriever_interactive() -> None:
	if not RETRIEVER_SCRIPT.exists():
		print("Retriever script not found.")
		return
	run_command([sys.executable, str(RETRIEVER_SCRIPT)], capture=False)


def run_labelling_interactive() -> None:
	if not LABELING_SCRIPT.exists():
		print("Labelling script not found.")
		return
	run_command([sys.executable, str(LABELING_SCRIPT)], capture=False)


def docker_up() -> bool:
	return run_command(["docker", "compose", "up", "-d", "--build"], capture=False)



def _port_open(host: str, port: int) -> bool:
	try:
		with socket.create_connection((host, port), timeout=2):
			return True
	except OSError:
		return False


def docker_reindex() -> bool:
	if not _port_open("127.0.0.1", 9200):
		print("ElasticSearch is not reachable on 127.0.0.1:9200.")
		print("Start services first or free port 9200.")
		return False
	return run_command([
		"docker",
		"compose",
		"run",
		"--no-deps",
		"--rm",
		"--workdir",
		"/app/retrieval",
		"retriever",
		"python",
		"run_retriever.py",
		"setup",
	], capture=False)


def docker_build_retriever() -> bool:
	if not DOCKERFILE_RETRIEVER.exists():
		print("Dockerfile.retriever not found.")
		return False
	return run_command([
		"docker",
		"build",
		"-t",
		IMAGE_TAG,
		"-f",
		str(DOCKERFILE_RETRIEVER),
		".",
	], capture=False)


def docker_menu() -> None:
	while True:
		print("\nDocker Services")
		print("a. Initialize and run ElasticSearch (and other images)")
		print("b. Create retriever Docker image (and update)")
		print("c. Reindex passages in ElasticSearch")
		print("d. Back")

		choice = input("\nSelect an option: ").strip().lower()
		if choice == "a":
			if docker_up():
				print("Docker services started.")
		elif choice == "b":
			if docker_build_retriever():
				print(f"Built image: {IMAGE_TAG}")
		elif choice == "c":
			if docker_reindex():
				print("Reindex complete.")
		elif choice == "d":
			break
		else:
			print("Invalid choice.")


def scheduler_menu() -> None:
	from scheduler.fastapi_who_scheduler import run_who_scraper
	from scheduler.fastapi_indexing_scheduler import run_indexing_updater

	while True:
		print("\nScheduling")
		print("1. Run WHO scraper (full)")
		print("2. Run WHO scraper (news)")
		print("3. Run WHO scraper (outbreak)")
		print("4. Run WHO scraper (features)")
		print("5. Run FAISS incremental indexing")
		print("6. Run FAISS full rebuild")
		print("7. Back")

		choice = input("\nSelect an option: ").strip().lower()
		if choice == "1":
			run_who_scraper(mode="full")
			print("WHO scraper completed.")
		elif choice == "2":
			run_who_scraper(mode="news")
			print("WHO scraper completed.")
		elif choice == "3":
			run_who_scraper(mode="outbreak")
			print("WHO scraper completed.")
		elif choice == "4":
			run_who_scraper(mode="features")
			print("WHO scraper completed.")
		elif choice == "5":
			run_indexing_updater(rebuild=False)
			print("FAISS incremental indexing completed.")
		elif choice == "6":
			run_indexing_updater(rebuild=True)
			print("FAISS full rebuild completed.")
		elif choice == "7":
			break
		else:
			print("Invalid choice.")


def main() -> None:
	while True:
		print("\nMain Menu")
		print("1. Run Retriever system")
		print("2. Labelling system")
		print("3. Docker Services")
		print("4. Scheduling")
		print("5. Exit")

		choice = input("\nSelect an option: ").strip().lower()
		if choice == "1":
			run_retriever_interactive()
		elif choice == "2":
			run_labelling_interactive()
		elif choice == "3":
			docker_menu()
		elif choice == "4":
			scheduler_menu()
		elif choice == "5":
			print("Goodbye.")
			break
		else:
			print("Invalid choice.")


if __name__ == "__main__":
	main()
