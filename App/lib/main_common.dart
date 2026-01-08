import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_native_splash/flutter_native_splash.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:verifact_app/flavors/flavor_config.dart';
import 'package:verifact_app/utils/constants/app_globals.dart';
import 'package:verifact_app/utils/constants/enums.dart';
import 'package:verifact_app/utils/constants/route_table.dart';
import 'package:verifact_app/utils/helpers/helper_functions.dart';

List<CameraDescription> cameras = [];

Future<void> mainCommon({
  required Flavor flavor,
  required String baseUrl,
  required String name,
}) async {
  // Initializing the Flavor config
  FlavorConfig(flavor: flavor, baseUrl: baseUrl, name: name);
  final widgetsBinding = WidgetsFlutterBinding.ensureInitialized();

  // Preserve the splash screen until initialization is complete
  FlutterNativeSplash.preserve(widgetsBinding: widgetsBinding);

  cameras = await availableCameras();

  // To force lock rotation of the app, uncommment the below line
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
  ]);

  // Remove the splash screen after initialization is complete
  FlutterNativeSplash.remove();

  runApp(const ProviderScope(child: MyApp()));
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  final ImagePicker _picker = ImagePicker();
  final TextRecognizer _textRecognizer = TextRecognizer();

  String _recognizedText = '';
  XFile? _imageFile;
  bool _isProcessing = false;
  File? selectedImage;

  @override
  void dispose() {
    _textRecognizer.close();
    super.dispose();
  }

  /// `Request permission for camera or gallery access.`
  Future<bool> _requestPermissionFor(ImageSource source) async {
    if (source == ImageSource.camera) {
      var status = await Permission.camera.status;
      debugPrint('[permission] camera status: $status');
      if (status.isDenied) {
        status = await Permission.camera.request();
        debugPrint('[permission] camera requested, new status: $status');
      }
      if (status.isPermanentlyDenied) {
        // Let the caller show an explanatory dialog and offer to open settings.
        return false;
      }
      return status.isGranted;
    } else {
      // gallery/photos
      if (Platform.isIOS) {
        var status = await Permission.photos.status;
        debugPrint('ℹ️ [permission] photos status: $status');
        if (status.isDenied) {
          status = await Permission.photos.request();
          debugPrint('ℹ️ [permission] photos requested, new status: $status');
        }
        // iOS 14+ has a 'limited' status; treat it as allowed for reading selected photos.
        if (status == PermissionStatus.limited) return true;
        // If the user permanently denied Photos permission, the modern iOS photo picker
        // (PHPicker) may still present a picker without granting the Photos permission to
        // the app. To avoid blocking the user from selecting an image, allow proceeding
        // to the picker. If you want to force the user to re-enable permissions, the
        // caller will show an explicit dialog to open App Settings.
        if (status.isPermanentlyDenied) return true;
        return status.isGranted;
      } else {
        var imageStatus = await Permission.photos.status;
        var extStorage = await Permission.manageExternalStorage.status;

        // Android: request storage (legacy) permission as a fallback
        var status = await Permission.storage.status;
        if (status.isDenied || imageStatus.isDenied || extStorage.isDenied) {
          status = await Permission.storage.request();
          imageStatus = await Permission.photos.request();
          extStorage = await Permission.manageExternalStorage.request();
        }
        if (status.isPermanentlyDenied ||
            imageStatus.isPermanentlyDenied ||
            extStorage.isPermanentlyDenied) {
          return false;
        }
        return imageStatus.isGranted ||
            status.isGranted ||
            extStorage.isGranted;
      }
    }
  }

  /// `Pick an image from gallery or camera, then recognize text.`
  Future<void> _pickImage(ImageSource source) async {
    if (!(Platform.isIOS && source == ImageSource.gallery)) {
      final granted = await _requestPermissionFor(source);
      if (!granted) {
        if (!mounted) return;
        await Permission.photos.request();
        _showPermissionDeniedDialog();
        return;
      }
    }

    try {
      setState(() {
        _isProcessing = true;
        _recognizedText = '';
      });

      final pickedFile = await _picker
          .pickImage(source: source, maxWidth: 2048, imageQuality: 85)
          .timeout(
            const Duration(seconds: 20),
            onTimeout: () {
              debugPrint('[picker] pickImage timed out after 20s');
              return null;
            },
          );
      debugPrint('[picker] pickImage returned: ${pickedFile?.path}');

      if (pickedFile == null) return;

      final inputImage = InputImage.fromFilePath(pickedFile.path);
      final recognizedText = await _textRecognizer.processImage(
        inputImage,
      );

      setState(() {
        _imageFile = pickedFile;
        _recognizedText = recognizedText.text;
      });
    } catch (e) {
      debugPrint('Error picking image: $e');
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Failed to pick image: $e')));
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  /// `Show a dialog informing the user that permission is required.`
  void _showPermissionDeniedDialog() {
    showErrorSnackbar('Permission Denied');
    // showDialog(
    //   context: context,
    //   builder: (context) => AlertDialog(
    //     title: const Text('Permission Required'),
    //     content: const Text(
    //       'This feature requires permission. You can grant it in Settings.',
    //     ),
    //     actions: [
    //       TextButton(
    //         onPressed: () => Navigator.of(context).pop(),
    //         child: const Text('Cancel'),
    //       ),
    //       TextButton(
    //         onPressed: () async {
    //           await openAppSettings();
    //           Navigator.of(context).pop();
    //         },
    //         child: const Text('Open Settings'),
    //       ),
    //     ],
    //   ),
    // );
  }

  @override
  Widget build(BuildContext context) {
    Route<dynamic>? onGenerateRoute(RouteSettings routeSettings) {
      final name = routeSettings.name;
      final pageBuilder = (name != null) ? appRoutes[name] : null;

      // If we found a builder in the table, build a platform-appropriate route.
      if (pageBuilder != null) {
        // Get the current platform in a way safe for all targets (web included).
        final platform = Theme.of(context).platform;

        // Build the route based on the platform
        switch (platform) {
          case TargetPlatform.iOS:
          case TargetPlatform.macOS:
            return CupertinoPageRoute(
              builder: pageBuilder,
              settings: routeSettings,
            );

          default:
            // MaterialPageRoute will respect ThemeData.pageTransitionsTheme (so your
            // CustomPredictiveBackTransitionBuilder gets used on Android if configured).
            return MaterialPageRoute(
              builder: pageBuilder,
              settings: routeSettings,
            );
        }
      }
      return MaterialPageRoute(
        builder: (_) => Scaffold(
          appBar: AppBar(title: const Text('Route not found')),
          body: Center(
            child: Text('No route defined for ${routeSettings.name}'),
          ),
        ),
        settings: routeSettings,
      );
    }

    return ScreenUtilInit(
      designSize: const Size(393, 852),
      minTextAdapt: true,
      splitScreenMode: true,

      builder: (context, child) => MaterialApp(
        onUnknownRoute: (settings) => MaterialPageRoute(
          builder: (_) => Scaffold(
            appBar: AppBar(title: const Text('Unknown route')),
            body: Center(child: Text('No route defined for ${settings.name}')),
          ),
          settings: settings,
        ),
        title: 'Verifact',
        scaffoldMessengerKey: scaffoldMessengerKey,
        navigatorKey: navigatorKey,
        home: Scaffold(
          body: SafeArea(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  /// `Recognized Display Area`
                  Expanded(
                    child: _isProcessing
                        ? const Center(
                            child: CircularProgressIndicator.adaptive(),
                          )
                        : _imageFile == null
                        ? const Center(child: Text('No image selected'))
                        : SingleChildScrollView(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.stretch,
                              children: [
                                Image.file(File(_imageFile!.path)),
                                const SizedBox(height: 12),

                                /// `Recognized Text`
                                Container(
                                  padding: const EdgeInsets.all(12),
                                  color: Colors.black87,
                                  child: Text(
                                    _recognizedText.isEmpty
                                        ? 'No text recognized.'
                                        : _recognizedText,
                                    style: const TextStyle(color: Colors.white),
                                  ),
                                ),
                                const SizedBox(height: 12),

                                /// `Copy recognized text to clipboard`
                                ElevatedButton.icon(
                                  onPressed: _recognizedText.isEmpty
                                      ? null
                                      : () {
                                          Clipboard.setData(
                                            ClipboardData(
                                              text: _recognizedText,
                                            ),
                                          );
                                          if (!mounted) return;
                                          showInfoSnackbar(
                                            'Text copied to clipboard',
                                          );
                                        },
                                  icon: const Icon(Icons.copy),
                                  label: const Text('Copy Text'),
                                ),
                                const SizedBox(height: 52),
                              ],
                            ),
                          ),
                  ),

                  const SizedBox(height: 16),

                  /// `Action Buttons`
                  Align(
                    alignment: AlignmentGeometry.bottomCenter,
                    child: Wrap(
                      alignment: WrapAlignment.center,
                      spacing: 12,
                      runSpacing: 12,
                      children: [
                        /// `Gallery Button`
                        ElevatedButton.icon(
                          onPressed: () => _pickImage(ImageSource.gallery),
                          icon: const Icon(Icons.photo_library),
                          label: const Text('Pick Image'),
                        ),

                        /// `Camera Button`
                        ElevatedButton.icon(
                          onPressed: () => _pickImage(ImageSource.camera),
                          icon: const Icon(Icons.camera_alt),
                          label: const Text('Use Camera'),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// import 'package:flutter/material.dart';
// import 'package:verifact_app/utils/constants/enums.dart';

// void main() {
//   runApp(const MainApp());
// }

// class MainApp extends StatelessWidget {
//   const MainApp({super.key});

//   @override
//   Widget build(BuildContext context) {
//     return const MaterialApp(
//       home: Scaffold(body: Center(child: Text('Hello World!'))),
//     );
//   }
// }
