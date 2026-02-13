import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

/// Helper service that opens the device camera and can show the
/// captured image on a simple preview page.
class CameraService {
  CameraService._();

  static final ImagePicker _picker = ImagePicker();

  /// Launches the device camera and returns the picked file (or null).
  static Future<XFile?> takePhoto({int imageQuality = 85}) async {
    try {
      final XFile? file = await _picker.pickImage(
        source: ImageSource.camera,
        imageQuality: imageQuality,
      );
      return file;
    } catch (_) {
      return null;
    }
  }

  /// Launches the camera and, if a photo is taken, navigates to
  /// a preview page that displays the captured image.
  static Future<void> openCameraAndShow(BuildContext context) async {
    final file = await takePhoto();
    if (file == null) return;

    Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => _ImagePreviewScreen(imagePath: file.path),
      ),
    );
  }
}

class _ImagePreviewScreen extends StatelessWidget {
  const _ImagePreviewScreen({required this.imagePath});

  final String imagePath;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Photo Preview'),
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(12.0),
          child: Image.file(
            File(imagePath),
            fit: BoxFit.contain,
            width: double.infinity,
          ),
        ),
      ),
    );
  }
}
