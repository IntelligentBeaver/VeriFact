import 'dart:io';

import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

/// Requests camera & gallery/photo permissions on startup.
class PermissionService {
  PermissionService._();

  /// Request camera and gallery (photos/storage) permissions.
  /// Returns true if required permissions are granted.
  static Future<bool> requestCameraAndGalleryPermissions({
    BuildContext? context,
  }) async {
    final cameraStatus = await Permission.camera.request();

    PermissionStatus galleryStatus;
    if (Platform.isIOS) {
      galleryStatus = await Permission.photos.request();
    } else {
      // Android: request storage read permission. Newer Android versions
      // may require manageExternalStorage; we request the common storage
      // permission which is sufficient for picking images.
      galleryStatus = await Permission.storage.request();
    }

    final granted = cameraStatus.isGranted && galleryStatus.isGranted;

    if (!granted && context != null) {
      // If denied forever, offer to open app settings
      final permanentlyDenied =
          cameraStatus.isPermanentlyDenied || galleryStatus.isPermanentlyDenied;

      if (permanentlyDenied) {
        await _showOpenSettingsDialog(context);
      }
    }

    return granted;
  }

  static Future<void> _showOpenSettingsDialog(BuildContext context) async {
    await showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Permissions required'),
        content: const Text(
          'Camera and gallery permissions are required to take and select photos.\n'
          'Please enable them in app settings.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () async {
              Navigator.of(ctx).pop();
              await openAppSettings();
            },
            child: const Text('Open Settings'),
          ),
        ],
      ),
    );
  }
}
