# VeriFact App

VeriFact is a Flutter application for scanning, extracting, and verifying text from images and documents using on-device ML and camera features. It provides OCR, capture workflows, result verification, history storage, and developer utilities for testing and debugging.

**Project snapshot**
- **SDK:** Dart/Flutter (environment: >= 3.10.1)
- **Flavors:** `dev`, `prod` (entrypoints: `lib/main_dev.dart`, `lib/main_prod.dart`)

**Key Features**
- Image capture and selection (camera + gallery)
- On-device OCR using Google ML Kit
- Document/text verification and result presentation
- Local history storage of scans
- Speech-to-text input and developer test screens
- Responsive UI and theming, with splash & launcher icon support

**Screens & Functionality**
- **Splash Screen:** App startup and initialization.
- **Home Screen:** Main entry UI — capture images, select images, start OCR workflows.
- **OCR Page:** Runs text recognition on image input and returns extracted text.
- **Verifier Result Screen:** Displays parsed text and verification outcome, confidence metadata, and actions (share, save).
- **History Screen:** Lists previously scanned/verified items stored locally.
- **History Preview Screen:** Detailed view of a saved scan with extracted text and metadata.
- **Settings Screen:** App preferences, permission toggles, and environment/config settings.
- **Response Test Screen:** Development/testing UI for API/response validation and diagnostics.
- **About Screen:** App info, version and credits.
- **App entry (`app.dart`):** Application root, routing and provider initialization.

Note: Screen filenames live in `lib/screens/` and the project also contains `lib/camera_screen/`, `lib/controllers/`, `lib/providers/`, `lib/services/`, `lib/models/`, and `lib/widgets/` to organize camera, state management, services, and UI components.

**Important Packages Used**
- `google_mlkit_text_recognition`: On-device OCR and text recognition.
- `camera` / `image_picker`: Capture images from camera and select images from gallery.
- `dio`: HTTP client for API requests and verification endpoints.
- `flutter_riverpod` (and `riverpod_annotation`): Primary state management and dependency injection.
- `provider`: Present for compatibility or legacy parts of the app.
- `hive` / `hive_flutter` / `shared_preferences`: Local persistence and lightweight storage for history and preferences.
- `speech_to_text`: Voice input and speech-to-text features.
- `permission_handler`: Manage runtime permissions for camera, storage and microphone.
- `connectivity_plus` / `device_info_plus`: Network and device capability checks.
- `flutter_dotenv`: Environment configuration via `.env` file.
- `flutter_screenutil`: Responsive layouts and sizing across screen densities.
- `flutter_native_splash` / `flutter_launcher_icons`: Splash screen and app icon generation.
- `logger`: Simple logging for development and debugging.
- `flutter_svg`, `google_fonts`: Asset & typography support.

**Project Structure (high level)**
- `lib/` — application source
	- `main_dev.dart`, `main_prod.dart`, `main_common.dart` — flavor entrypoints
	- `screens/` — UI screens (listed above)
	- `camera_screen/`, `controllers/`, `providers/`, `services/`, `models/`, `widgets/` — feature code
- `android/`, `ios/` — platform configurations and keystore/settings
- `assets/` — images, icons, fonts
- `tools/bump_version.py` — version bump helper (used by build tasks)

**Setup & Run**
Prerequisites: Flutter SDK (matching the project's environment), Android/iOS toolchains as needed.

Install dependencies:
```bash
flutter pub get
```

Run (dev flavor):
```bash
flutter run -t lib/main_dev.dart --flavor dev
```

Build (prod appbundle):
```bash
# optionally bump version first (workspace tasks exist that call the bump scripts)
flutter build appbundle --flavor prod -t lib/main_prod.dart --release
```

The workspace includes VS Code tasks for version bumping and building (see tasks named like `flutter: bump patch (prod)` and `flutter: build appbundle (prod) [patch bump]`). Version bumps use `tools/bump_version.py`.

**Configuration & Notes**
- Add runtime configuration in `.env` (the app loads `.env` per `pubspec.yaml`).
- App permissions must be granted for camera, microphone (speech) and storage for full functionality.
- Local persistence uses Hive; if schema or adapters are added, ensure `build_runner` is used to regenerate code where necessary.

**Contributing**
- Follow existing style and Riverpod-driven state patterns when adding features.
- Run `flutter analyze` and `dart fix --apply` before submitting changes.

**License**
This repository does not include a license file. Add a license if you intend to make this project public.

---
Updated documentation for the project's structure and developer usage.
