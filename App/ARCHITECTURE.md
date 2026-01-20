# 📊 OCR Page Architecture Diagram

## Project Structure

```
lib/
├── screens/
│   └── ocr_page.dart ........................ ✨ REFACTORED UI Layer
│       └── OcrScreen (ConsumerWidget)
│           ├── _buildDisplayArea()
│           ├── _buildActionButtons()
│           └── _buildPickerButtons()
│
├── providers/
│   └── ocr_provider.dart ................... ✨ NEW State Management
│       ├── OcrState (Immutable)
│       ├── OcrNotifier (Notifier)
│       └── Providers:
│           ├── ocrServiceProvider
│           ├── ocrNotifierProvider
│           ├── isOcrProcessingProvider
│           ├── hasRecognizedTextProvider
│           ├── recognizedTextProvider
│           └── ocrErrorProvider
│
├── services/
│   └── ocr_service.dart ................... ✨ NEW Business Logic
│       ├── OcrService
│       │   ├── pickImageAndRecognizeText()
│       │   ├── pickImage()
│       │   ├── recognizeText()
│       │   ├── requestPermission()
│       │   ├── hasPermission()
│       │   └── dispose()
│       │
│       ├── OcrPickerResult (Data Class)
│       └── Permission Handlers:
│           ├── _requestCameraPermission()
│           ├── _requestGalleryPermission()
│           ├── _requestGalleryPermissionIOS()
│           └── _requestGalleryPermissionAndroid()
│
└── utils/
    └── helpers/
        └── helper_functions.dart (existing)
            └── showInfoSnackbar() [used by OCR]
```

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         OcrScreen (UI)                       │
│              (Responsive - Watches Providers)                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├──► ref.watch(ocrNotifierProvider)
                     │         │
                     │         └─► OcrState (current)
                     │
                     └──► ref.read(ocrNotifierProvider.notifier)
                              │
                              └─► Call Methods:
                                  ├─ pickAndRecognizeImage()
                                  ├─ clearError()
                                  ├─ retakeImage()
                                  └─ reset()
                              
                                       │
                                       ▼
                    ┌──────────────────────────────────┐
                    │       OcrNotifier                │
                    │   (State Management Logic)       │
                    └──────────────────────────────────┘
                                       │
                                       ├──► Watches ocrServiceProvider
                                       │
                                       ▼
                    ┌──────────────────────────────────┐
                    │       OcrService                 │
                    │   (Business Logic & Permissions) │
                    └──────────────────────────────────┘
                              │
                              ├──► ImagePicker (gallery/camera)
                              ├──► TextRecognizer (ML Kit)
                              ├──► Permission Handler
                              └──► File System
                                       │
                                       ▼
                    ┌──────────────────────────────────┐
                    │    Result or Error               │
                    │  - XFile? imageFile              │
                    │  - String recognizedText         │
                    │  - Exception error               │
                    └──────────────────────────────────┘
                                       │
                                       ▼ (Caught by OcrNotifier)
                    ┌──────────────────────────────────┐
                    │    Update OcrState via           │
                    │    state.copyWith()              │
                    └──────────────────────────────────┘
                                       │
                                       ▼ (Notified to UI)
                    ┌──────────────────────────────────┐
                    │    UI Rebuilds Automatically     │
                    │   - Shows image or error         │
                    │   - Disables buttons if loading  │
                    │   - Shows error message          │
                    └──────────────────────────────────┘
```

---

## State Management Flow

```
Initial State:
┌────────────────────────────────────────┐
│  OcrState(                              │
│    imageFile: null,                    │
│    recognizedText: '',                 │
│    isProcessing: false,                │
│    error: null,                        │
│    isPermissionDenied: false           │
│  )                                     │
└────────────────────────────────────────┘

User clicks "Pick Image" ──────────────────────────┐
                                                    │
                                                    ▼
Processing State (Optimistic Update):
┌────────────────────────────────────────┐
│  OcrState(                              │
│    imageFile: null,  ◄─── keeping old   │
│    recognizedText: '',                 │
│    isProcessing: true,   ◄─── NEW!     │
│    error: null,          ◄─── clear    │
│    isPermissionDenied: false            │
│  )                                     │
└────────────────────────────────────────┘
UI shows: Loading spinner

Success Path:
                                    ▼
Processing → Service returns OcrPickerResult
                                    ▼
┌────────────────────────────────────────┐
│  OcrState(                              │
│    imageFile: XFile(...), ◄─── NEW!    │
│    recognizedText: '...', ◄─── NEW!    │
│    isProcessing: false,   ◄─── DONE    │
│    error: null,                        │
│    isPermissionDenied: false            │
│  )                                     │
└────────────────────────────────────────┘
UI shows: Image + Text

Error Path:
                                    ▼
Processing → Service throws Exception
                                    ▼
┌────────────────────────────────────────┐
│  OcrState(                              │
│    imageFile: null,                    │
│    recognizedText: '',                 │
│    isProcessing: false,   ◄─── DONE    │
│    error: 'Failed to process...', ◄─── NEW!
│    isPermissionDenied: false            │
│  )                                     │
└────────────────────────────────────────┘
UI shows: Error banner

Permission Denied Path:
                                    ▼
Processing → Service detects PERMISSION_DENIED
                                    ▼
┌────────────────────────────────────────┐
│  OcrState(                              │
│    imageFile: null,                    │
│    recognizedText: '',                 │
│    isProcessing: false,   ◄─── DONE    │
│    error: 'Permission denied...', ◄─── NEW!
│    isPermissionDenied: true  ◄─── NEW! │
│  )                                     │
└────────────────────────────────────────┘
UI shows: Error banner with setting guidance
```

---

## Component Interaction Diagram

```
                    ┌─────────────────┐
                    │   OcrScreen     │
                    │  (UI Layer)     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Providers     │
                    │  (Integration)  │
                    └────────┬────────┘
                             │
                  ┌──────────┴──────────┐
                  │                    │
          ┌───────▼──────────┐    ┌───▼──────────────┐
          │  OcrNotifier     │    │  OcrService      │
          │  (State)         │    │  (Logic)         │
          │                  │    │                  │
          │ - state          │    │ - ImagePicker    │
          │ - copyWith()     │    │ - TextRecognizer │
          │ - pickAnd...()   │◄───┤ - Permissions    │
          │ - clearError()   │    │ - File Handling  │
          │ - retakeImage()  │    │                  │
          │ - reset()        │    │                  │
          │ - dispose()      │    └──────────────────┘
          └──────────────────┘
```

---

## Permission Flow Diagram

```
User Action: Pick Image
    │
    ▼
hasPermission(source)?
    │
    ├─ YES ────────────────────────► ShowPicker
    │
    └─ NO
        │
        ▼
    Is iOS + Gallery? ──── YES ────► ShowPicker (PHPicker handles it)
    │
    └─ NO
        │
        ▼
    requestPermission(source)
        │
        ├─ GRANTED ──────────────────► ShowPicker
        │
        ├─ DENIED
        │   │
        │   └──────────────────────► ShowError "Permission Denied"
        │
        └─ PERMANENTLY DENIED
            │
            └──────────────────────► ThrowException "PERMISSION_DENIED"
                                      │
                                      ▼ (Caught by OcrNotifier)
                                      
                                    Show Error Banner:
                                    "Permission denied.
                                     Please enable access in settings."
```

---

## Platform-Specific Behavior

```
                    ┌──────────────────┐
                    │  Image Source    │
                    └────────┬─────────┘
                             │
                    ┌────────┴────────┐
                    │                │
         ┌──────────▼──────────┐  ┌──▼─────────────────┐
         │  ImageSource.Camera │  │ImageSource.Gallery │
         └──────────┬──────────┘  └────────┬───────────┘
                    │                      │
                    ▼                      ▼
         ┌─────────────────────┐  ┌──────────────────┐
         │ Camera Permission   │  │ Platform Check   │
         │                     │  └────────┬─────────┘
         │ Request:            │           │
         │ Permission.camera   │   ┌───────┴────────┐
         │                     │   │                │
         │ iOS & Android:      │┌──▼──────┐   ┌────▼─────┐
         │ Same behavior       ││  iOS    │   │ Android  │
         └─────────────────────┘│         │   │          │
                                │ Photo   │   │ Photos   │
                                │ Status: │   │ Status:  │
                                │ - granted
                                │ - limited   │ - granted
                                │ - denied    │ - denied
                                │ - perm.     │ - perm.
                                │   denied    │   denied
                                │         │   │          │
                                │ Action: │   │ Action:  │
                                │ - Try   │   │ - Try    │
                                │   PHPicker  │ storage  │
                                │ - on error  │ & photos │
                                └─────────┘   └──────────┘
```

---

## Logging & Debugging Points

```
OcrService                          OcrNotifier              OcrScreen
    │                                   │                        │
    ├─ [OCR Permission]                 │                        │
    │  camera/photos status check       │                        │
    │                                   │                        │
    ├─ [OCR Picker]                     │                        │
    │  pickImage timed out              │                        │
    │                                   │                        │
    ├─ [OCR Recognition]                │                        │
    │  Recognized X characters          │                        │
    │                                   │                        │
    └─ [OCR Service]                    │                        │
       Error in operation               │                        │
                                        │                        │
                                        ├─ State changed         │
                                        │  isProcessing: true    │
                                        │                        │
                                        ├─ State changed         │
                                        │  error updated         │
                                        │                        │
                                        └─ State changed         │
                                           imageFile & text      │
                                                                  │
                                                                  ├─ UI rebuilds
                                                                  │
                                                                  ├─ Shows spinner
                                                                  │  or error
                                                                  │  or image
                                                                  │
                                                                  └─ User sees
                                                                     result
```

---

## Class Responsibility Matrix

| Class | Responsibility | Knows About | Doesn't Know About |
|-------|---|---|---|
| **OcrScreen** | Render UI | OcrNotifier, OcrState | Network, File I/O |
| **OcrNotifier** | State management | OcrService, OcrState | UI, Widgets |
| **OcrService** | Business logic | ImagePicker, TextRecognizer, Permissions | Riverpod, UI |
| **OcrState** | Data container | Its own properties | Logic, Services |

---

## Testing Strategy

```
Unit Tests:
├─ OcrService
│  ├─ pickImage() behavior
│  ├─ recognizeText() behavior
│  ├─ Permission request logic
│  └─ Error handling
│
├─ OcrNotifier
│  ├─ State updates
│  ├─ Error handling
│  └─ State reset
│
└─ OcrState
   └─ copyWith() functionality

Widget Tests:
├─ OcrScreen
│  ├─ Loading state display
│  ├─ Empty state display
│  ├─ Error state display
│  ├─ Image display
│  ├─ Button interactions
│  └─ Error dismissal

Integration Tests:
├─ Full image pick flow
├─ Text recognition flow
└─ Permission request flow
```

---

## Summary

**The refactored OCR page follows a clean, layered architecture:**

1. **UI Layer** - Reactive widgets using Riverpod
2. **State Layer** - Immutable state with Notifier pattern
3. **Service Layer** - Reusable business logic
4. **Data Layer** - Simple data classes

**Each layer has a single responsibility and can be tested independently.**

This architecture scales well and serves as a template for other screens in your app!
