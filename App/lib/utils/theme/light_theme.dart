import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:verifact_app/utils/constants/colors.dart';

/// Get the value of Light theme of type [ThemeData]
ThemeData get lightTheme => _lightTheme;

/// Get the value of Color scheme of type [ColorScheme]
ColorScheme get colorScheme => _lightColorScheme;

ThemeData _lightTheme = ThemeData(
  brightness: Brightness.light,
  colorScheme: _lightColorScheme,
  scaffoldBackgroundColor: AppColors.background,
  primaryColor: AppColors.primary,
  splashColor: AppColors.tile,
  highlightColor: Colors.transparent,
  hoverColor: AppColors.tile,
  disabledColor: AppColors.primaryDisabled,
  fontFamily: GoogleFonts.poppins().fontFamily,

  // Custom Themes Defined
  // chipTheme: AppChipTheme.lightTheme,
  // dialogTheme: AppDialogTheme.lightTheme,
  // pageTransitionsTheme: AppPageTransitionsTheme.pageTransitionsTheme,
  // dropdownMenuTheme: AppDropdownMenuTheme.dropdownMenuThemeLight,
  // inputDecorationTheme: AppInputDecoration.lightTheme,
  // snackBarTheme: AppSnackerBarTheme.lightTheme,
  // listTileTheme: AppListTileTheme.lightTheme,
  // textTheme: AppTypography.textTheme,
  // switchTheme: AppSwitchTheme.lightTheme,
  // cardTheme: AppCardTheme.lightTheme,
  // checkboxTheme: AppCheckBoxTheme.lightTheme,
  // bottomSheetTheme: AppBottomSheetTheme.lightTheme,
  // progressIndicatorTheme: AppProgressIndicatorTheme.lightTheme,
  // searchBarTheme: AppSearchBarTheme.lightTheme,

  //* App and Tab Bar Themes
  // appBarTheme: CustomAppBarTheme.lightTheme,
  // tabBarTheme: CustomTabBarTheme.lightTheme,

  //* Button Themes
  // elevatedButtonTheme: AppButtonTheme.lightTheme,
  // iconTheme: AppIconTheme.lightTheme,
  // filledButtonTheme: AppFilledButtonTheme.lightTheme,
  // outlinedButtonTheme: AppOutlinedButtonTheme.lightTheme,
);

ColorScheme _lightColorScheme = const ColorScheme.light(
  // Primary
  primary: AppColors.primary,
  inversePrimary: AppColors.primaryDark,

  // Secondary
  secondary: AppColors.secondary,
  onSecondary: AppColors.onSecondary,

  // Tertiary
  tertiary: AppColors.tertiary,
  onTertiary: AppColors.onTertiary,

  inverseSurface: AppColors.surfaceDark,

  // Container
  primaryContainer: AppColors.primaryBackground,

  // Error
  error: AppColors.error,
  errorContainer: AppColors.errorContainer,
  onErrorContainer: AppColors.onErrorContainer,
);
