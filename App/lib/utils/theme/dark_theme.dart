import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:verifact_app/utils/constants/colors.dart';

/// Get the value of Light theme of type [ThemeData]
ThemeData get darkTheme => _darkTheme;

/// Get the value of Color scheme of type [ColorScheme]
ColorScheme get colorScheme => _darkColorScheme;

ThemeData _darkTheme = ThemeData(
  brightness: Brightness.dark,
  colorScheme: _darkColorScheme,
  scaffoldBackgroundColor: AppColors.backgroundDark,
  primaryColor: AppColors.primaryDark,
  splashColor: AppColors.tileDark,
  highlightColor: Colors.transparent,
  hoverColor: AppColors.tileDark,
  disabledColor: AppColors.primaryDisabledDark,
  fontFamily: GoogleFonts.poppins().fontFamily,

  // Custom Themes Defined
  // pageTransitionsTheme: AppPageTransitionsTheme.pageTransitionsTheme,
  // dropdownMenuTheme: AppDropdownMenuTheme.dropdownMenuThemeDark,
  // textTheme: AppTypography.textTheme,
  // inputDecorationTheme: AppInputDecoration.darkTheme,
  // snackBarTheme: AppSnackerBarTheme.darkTheme,
  // listTileTheme: AppListTileTheme.darkTheme,
  // switchTheme: AppSwitchTheme.darkTheme,
  // dialogTheme: AppDialogTheme.darkTheme,
  // chipTheme: AppChipTheme.darkTheme,
  // cardTheme: AppCardTheme.darkTheme,
  // checkboxTheme: AppCheckBoxTheme.darkTheme,
  // bottomSheetTheme: AppBottomSheetTheme.darkTheme,
  // progressIndicatorTheme: AppProgressIndicatorTheme.darkTheme,
  // searchBarTheme: AppSearchBarTheme.darkTheme,

  //* App and Tab Bar Themes
  // appBarTheme: CustomAppBarTheme.darkTheme,
  // tabBarTheme: CustomTabBarTheme.darkTheme,

  //* Button Themes
  // outlinedButtonTheme: AppOutlinedButtonTheme.darkTheme,
  // filledButtonTheme: AppFilledButtonTheme.darkTheme,
  // elevatedButtonTheme: AppButtonTheme.darkTheme,
  iconTheme: const IconThemeData(
    color: AppColors.textPrimary,
    applyTextScaling: true,
  ),
);

ColorScheme _darkColorScheme = const ColorScheme.dark(
  // Primary
  primary: AppColors.primaryDark,
  onPrimary: AppColors.onPrimaryDark,
  inversePrimary: AppColors.primary,

  // Secondary
  secondary: AppColors.secondaryDark,
  onSecondary: AppColors.onSecondaryDark,

  // Tertiary
  tertiary: AppColors.tertiaryDark,
  onTertiary: AppColors.onTertiaryDark,

  // Surface
  surface: AppColors.surfaceDark,
  inverseSurface: AppColors.surface,

  // Error
  error: AppColors.errorDark,
);
