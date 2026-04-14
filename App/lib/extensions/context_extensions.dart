import 'package:flutter/material.dart';
import 'package:verifact_app/utils/helpers/device_utility.dart';

/// Extension on [BuildContext] providing shorthand methods for localization,
/// theme access, and gradients.
extension AppContextExtension on BuildContext {
  /// A method tr() for shorthand of AppLocalizations.of(this)!.translate(key)
  // String tr(String key) {
  //   return AppLocalizations.of(this)!.translate(key);
  // }

  // -------------------------------
  // Gradient shortcuts
  // -------------------------------
  /// Returns a primary vertical gradient, adapted to dark mode
  LinearGradient get gradient {
    final isDarkMode = DeviceUtility.isDarkMode(this);

    return LinearGradient(
      begin: Alignment.topCenter,
      end: Alignment.bottomCenter,
      colors: [
        Colors.deepPurple.shade200,
        Colors.deepPurple.shade300,
        if (isDarkMode) Colors.black else Colors.white,
      ],
      stops: const [0.0, 0.2, 0.6],
    );
  }
}

/// Extension on [BuildContext] to provide concise access to common [ThemeData] properties.
///
/// Use this to avoid repeatedly writing `Theme.of(context)` and to improve readability.
extension ThemeContextExtension on BuildContext {
  /// Returns the current [ThemeData] from the widget tree.
  ///
  /// Equivalent to `Theme.of(context)`.
  ThemeData get theme => Theme.of(this);

  /// Returns the current [ColorScheme] from the theme.
  ///
  /// Equivalent to `Theme.of(context).colorScheme`.
  ColorScheme get color => Theme.of(this).colorScheme;

  /// Returns the current [TextTheme] from the theme.
  ///
  /// Equivalent to `Theme.of(context).textTheme`.
  TextTheme get text => Theme.of(this).textTheme;
}
