import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:verifact_app/utils/constants/colors.dart';
import 'package:verifact_app/utils/theme/text_theme.dart';

/// Get the value of Dark theme of type [ThemeData]
ThemeData get darkTheme => _darkTheme;

/// Get the value of Color scheme of type [ColorScheme]
ColorScheme get darkColorScheme => _darkColorScheme;

final ColorScheme _darkColorScheme =
    ColorScheme.fromSeed(
      seedColor: AppColors.primary,
      brightness: Brightness.dark,
    ).copyWith(
      primary: AppColors.primary,
      onPrimary: AppColors.onPrimary,
      primaryContainer: AppColors.primaryContainerDark,
      onPrimaryContainer: AppColors.onPrimaryContainerDark,
      secondary: AppColors.secondary,
      onSecondary: AppColors.onSecondary,
      secondaryContainer: AppColors.secondaryContainerDark,
      onSecondaryContainer: AppColors.onSecondaryContainerDark,
      tertiary: AppColors.tertiary,
      onTertiary: AppColors.onTertiary,
      tertiaryContainer: AppColors.tertiaryContainerDark,
      onTertiaryContainer: AppColors.onTertiaryContainerDark,
      error: AppColors.error,
      onError: AppColors.onError,
      errorContainer: AppColors.errorContainerDark,
      onErrorContainer: AppColors.onErrorContainerDark,
      surface: AppColors.surfaceDark,
      onSurface: AppColors.textPrimaryDark,
      surfaceContainerHighest: AppColors.surfaceVariantDark,
      onSurfaceVariant: AppColors.textSecondaryDark,
      outline: AppColors.outlineDark,
      outlineVariant: AppColors.outlineVariantDark,
      shadow: AppColors.shadow,
      scrim: AppColors.scrim,
      inverseSurface: AppColors.inverseSurfaceDark,
      onInverseSurface: AppColors.onInverseSurfaceDark,
      inversePrimary: AppColors.primaryDark,
      surfaceTint: AppColors.primary,
    );

final ThemeData _darkTheme = ThemeData(
  useMaterial3: true,
  brightness: Brightness.dark,
  colorScheme: _darkColorScheme,
  scaffoldBackgroundColor: AppColors.backgroundDark,
  primaryColor: AppColors.primary,
  splashColor: AppColors.primaryHoverDark,
  highlightColor: Colors.transparent,
  hoverColor: AppColors.primaryHoverDark,
  disabledColor: AppColors.disabledDark,
  fontFamily: GoogleFonts.manrope().fontFamily,
  textTheme: AppTypography.darkTheme,
  iconTheme: const IconThemeData(color: AppColors.iconPrimaryDark),
  appBarTheme: const AppBarTheme(
    backgroundColor: AppColors.surfaceDark,
    foregroundColor: AppColors.textPrimaryDark,
    surfaceTintColor: Colors.transparent,
    elevation: 0,
  ),
);
