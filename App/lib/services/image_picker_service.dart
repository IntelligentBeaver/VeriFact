import 'dart:io';

import 'package:image_picker/image_picker.dart';

/// Simple wrapper around `image_picker` to centralize image selection.
class ImagePickerService {
  ImagePickerService._();

  static final ImagePicker _picker = ImagePicker();

  /// Pick an image from the device gallery. Returns an [XFile] or null.
  static Future<XFile?> pickFromGallery({int imageQuality = 85}) async {
    try {
      final XFile? file = await _picker.pickImage(
        source: ImageSource.gallery,
        imageQuality: imageQuality,
      );
      return file;
    } catch (_) {
      return null;
    }
  }

  /// Pick an image using the camera. Returns an [XFile] or null.
  static Future<XFile?> pickFromCamera({int imageQuality = 85}) async {
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

  /// Convenience: get a [File] from gallery selection (or null).
  static Future<File?> pickFileFromGallery({int imageQuality = 85}) async {
    final xfile = await pickFromGallery(imageQuality: imageQuality);
    if (xfile == null) return null;
    return File(xfile.path);
  }

  /// Convenience: get a [File] from camera capture (or null).
  static Future<File?> pickFileFromCamera({int imageQuality = 85}) async {
    final xfile = await pickFromCamera(imageQuality: imageQuality);
    if (xfile == null) return null;
    return File(xfile.path);
  }
}
