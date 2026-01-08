import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:verifact_app/camera_screen/camera_screen.dart';

final Map<String, WidgetBuilder> appRoutes = {
  CameraScreen.routeName: (context) => const CameraScreen(
    camera: CameraDescription(
      name: 'Default',
      lensDirection: CameraLensDirection.back,
      sensorOrientation: 0,
    ),
  ),

  
};
