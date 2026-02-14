#!/usr/bin/env python3
"""
Verification script for WHO scraping scheduler setup.
Run this to check if everything is configured correctly before running the scheduler.
"""

import sys
from pathlib import Path
from datetime import datetime

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_status(message, status="info"):
    """Print colored status message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if status == "success":
        print(f"{GREEN}✓{RESET} [{timestamp}] {message}")
    elif status == "error":
        print(f"{RED}✗{RESET} [{timestamp}] {message}")
    elif status == "warning":
        print(f"{YELLOW}⚠{RESET} [{timestamp}] {message}")
    else:
        print(f"{BLUE}ℹ{RESET} [{timestamp}] {message}")


def verify_paths():
    """Verify all file paths are correct."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Path Verification{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    issues = []
    
    # Get base paths
    script_dir = Path(__file__).parent
    adapters_dir = script_dir.parent.parent / "adapters" / "who" / "scraping"
    storage_dir = script_dir.parent.parent / "storage" / "who"
    
    print_status(f"Script directory: {script_dir}", "info")
    print_status(f"Adapters directory: {adapters_dir}", "info")
    print_status(f"Storage directory: {storage_dir}", "info")
    print()
    
    # Check if adapters directory exists
    if adapters_dir.exists():
        print_status(f"Adapters directory exists", "success")
    else:
        print_status(f"Adapters directory NOT found: {adapters_dir}", "error")
        issues.append(f"Adapters directory missing: {adapters_dir}")
    
    # Check individual adapter scripts
    required_scripts = {
        "News Headlines": "adapter_who_news_headlines.py",
        "News Details": "adapter_who_news_details.py",
        "Disease Outbreak Headlines": "adapter_who_disease_outbreak_headlines.py",
        "Disease Outbreak Details": "adapter_who_disease_outbreak_details.py",
        "Feature Stories Headlines": "adapter_who_feature_stories_headlines.py",
        "Feature Stories Details": "adapter_who_feature_stories_details.py",
    }
    
    print()
    for name, script in required_scripts.items():
        script_path = adapters_dir / script
        if script_path.exists():
            print_status(f"{name}: {script}", "success")
        else:
            print_status(f"{name}: {script} NOT FOUND", "error")
            issues.append(f"Missing script: {script}")
    
    # Check storage directories
    print()
    storage_subdirs = [
        "headlines",
        "news",
        "disease_outbreak_headlines",
        "disease_outbreak_news",
        "feature_stories_headlines",
        "feature_stories_news",
    ]
    
    for subdir in storage_subdirs:
        subdir_path = storage_dir / subdir
        if subdir_path.exists():
            print_status(f"Storage: {subdir}/", "success")
        else:
            print_status(f"Storage: {subdir}/ (will be created)", "warning")
    
    return issues


def verify_imports():
    """Verify required Python packages are installed."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Package Verification{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    issues = []
    
    required_packages = {
        "requests": "requests",
        "bs4": "beautifulsoup4",
        "selenium": "selenium",
    }
    
    optional_packages = {
        "apscheduler": "apscheduler (for FastAPI integration)",
        "fastapi": "fastapi (for API integration)",
    }
    
    # Check required packages
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            print_status(f"{package_name}", "success")
        except ImportError:
            print_status(f"{package_name} NOT INSTALLED", "error")
            issues.append(f"Missing package: pip install {package_name}")
    
    print()
    
    # Check optional packages
    for module_name, package_name in optional_packages.items():
        try:
            __import__(module_name)
            print_status(f"{package_name}", "success")
        except ImportError:
            print_status(f"{package_name} (optional)", "warning")
    
    return issues


def verify_script_arguments():
    """Verify adapter scripts have required arguments."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Script Argument Verification{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    issues = []
    
    script_dir = Path(__file__).parent
    adapters_dir = script_dir.parent.parent / "adapters" / "who" / "scraping"
    
    # Check if scripts have required arguments by reading their content
    checks = [
        ("adapter_who_news_headlines.py", "--max-age-days", "News Headlines"),
        ("adapter_who_news_details.py", "--recent-days", "News Details"),
        ("adapter_who_disease_outbreak_headlines.py", "--max-age-days", "Outbreak Headlines"),
        ("adapter_who_disease_outbreak_details.py", "--min-year", "Outbreak Details"),
        ("adapter_who_feature_stories_headlines.py", "--max-age-days", "Feature Headlines"),
        ("adapter_who_feature_stories_details.py", "--recent-days", "Feature Details"),
    ]
    
    for script_name, arg_name, display_name in checks:
        script_path = adapters_dir / script_name
        if script_path.exists():
            try:
                content = script_path.read_text(encoding='utf-8')
                if arg_name in content:
                    print_status(f"{display_name}: has {arg_name}", "success")
                else:
                    print_status(f"{display_name}: missing {arg_name}", "error")
                    issues.append(f"{script_name} missing argument: {arg_name}")
            except Exception as e:
                print_status(f"{display_name}: error reading file - {e}", "error")
                issues.append(f"Cannot read {script_name}")
        else:
            print_status(f"{display_name}: script not found", "error")
    
    return issues


def verify_python_version():
    """Check Python version."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Python Environment{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    issues = []
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print_status(f"Python version: {version_str}", "info")
    print_status(f"Python executable: {sys.executable}", "info")
    
    if version.major >= 3 and version.minor >= 7:
        print_status(f"Python version OK (>= 3.7)", "success")
    else:
        print_status(f"Python version too old (need >= 3.7)", "error")
        issues.append("Python version must be >= 3.7")
    
    return issues


def test_scheduler_dry_run():
    """Test if the scheduler can be imported and run."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Scheduler Test{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    issues = []
    
    try:
        # Try importing the scheduler module
        script_dir = Path(__file__).parent
        sys.path.insert(0, str(script_dir))
        
        import who_scheduler
        print_status("Scheduler module imports successfully", "success")
        
        # Check if SCRAPERS dict is properly defined
        if hasattr(who_scheduler, 'SCRAPERS'):
            print_status("SCRAPERS configuration found", "success")
            
            # Verify all paths in SCRAPERS exist
            for category, scripts in who_scheduler.SCRAPERS.items():
                for script_type, script_path in scripts.items():
                    if script_path.exists():
                        print_status(f"  {category}/{script_type}: OK", "success")
                    else:
                        print_status(f"  {category}/{script_type}: NOT FOUND", "error")
                        issues.append(f"Script not found: {script_path}")
        else:
            print_status("SCRAPERS configuration missing", "error")
            issues.append("SCRAPERS dict not found in who_scheduler.py")
            
    except Exception as e:
        print_status(f"Error importing scheduler: {e}", "error")
        issues.append(f"Cannot import scheduler: {e}")
    
    return issues


def main():
    """Run all verification checks."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}WHO Scraping Scheduler - Setup Verification{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    all_issues = []
    
    # Run all checks
    all_issues.extend(verify_python_version())
    all_issues.extend(verify_paths())
    all_issues.extend(verify_imports())
    all_issues.extend(verify_script_arguments())
    all_issues.extend(test_scheduler_dry_run())
    
    # Print summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Verification Summary{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    if not all_issues:
        print_status("All checks passed! Scheduler is ready to use.", "success")
        print()
        print(f"{GREEN}You can now run:{RESET}")
        print(f"  python who_scheduler.py --mode full")
        print()
    else:
        print_status(f"Found {len(all_issues)} issue(s):", "error")
        print()
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
        print()
        print(f"{YELLOW}Please fix these issues before running the scheduler.{RESET}")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
