import 'package:camera/camera.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_native_splash/flutter_native_splash.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:verifact_app/flavors/flavor_config.dart';
import 'package:verifact_app/screens/app.dart';
import 'package:verifact_app/services/permission_service.dart';
import 'package:verifact_app/utils/constants/enums.dart';

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

  // Request camera & gallery permissions up front so the app can
  // immediately access the device camera or pick images.
  await PermissionService.requestCameraAndGalleryPermissions();

  // To force lock rotation of the app, uncommment the below line
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
  ]);

  // Remove the splash screen after initialization is complete
  FlutterNativeSplash.remove();

  runApp(
    const ProviderScope(
      child: App(),
    ),
  );
}
