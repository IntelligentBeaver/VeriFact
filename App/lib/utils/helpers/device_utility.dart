import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class DeviceUtility {
  DeviceUtility._();
  static bool isDarkMode(BuildContext context) {
    return Theme.of(context).brightness == Brightness.dark;
  }

  static void hideKeyboard(BuildContext context) {
    FocusScope.of(context).requestFocus(FocusNode());
  }

  static bool isLandscapeOrientation(BuildContext context) {
    final viewInsets = View.of(context).viewInsets;
    return viewInsets.bottom == 0;
  }

  static bool isPortraitOrientation(BuildContext context) {
    final viewInsets = View.of(context).viewInsets;
    return viewInsets.bottom != 0;
  }

  static double getScreenHeight(BuildContext context) {
    final media = MediaQuery.of(context);
    return media.size.height - media.viewInsets.bottom;
  }

  static double getScreenWidth(BuildContext context) {
    return MediaQuery.of(context).size.width;
  }

  static double getPixelRatio(BuildContext context) {
    return MediaQuery.of(context).devicePixelRatio;
  }

  static double getStatusBarHeight(BuildContext context) {
    return MediaQuery.of(context).padding.top;
  }

  static double getBottomBarHeight(BuildContext context) {
    return MediaQuery.of(context).padding.bottom;
  }

  static double getScaleFactor(BuildContext context) {
    const standardSize = 360;
    final deviceWidth = getScreenWidth(context);
    final scaleFactor = deviceWidth / standardSize;

    return scaleFactor;
  }

  static Future<void> setOrientation(
    List<DeviceOrientation> orientations,
  ) async {
    await SystemChrome.setPreferredOrientations(orientations);
  }

  static void vibrate(Duration duration) {
    unawaited(HapticFeedback.vibrate());
    Future.delayed(duration, HapticFeedback.vibrate);
  }
}
