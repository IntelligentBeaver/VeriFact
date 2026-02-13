import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:verifact_app/utils/constants/colors.dart';
import 'package:verifact_app/utils/theme/text_theme.dart';

/// Get the value of Light theme of type [ThemeData]
ThemeData get lightTheme => _lightTheme;

/// Get the value of Color scheme of type [ColorScheme]
ColorScheme get lightColorScheme => _lightColorScheme;

final ColorScheme _lightColorScheme =
    ColorScheme.fromSeed(
      seedColor: AppColors.primary,
      brightness: Brightness.light,
    ).copyWith(
      primary: AppColors.primary,
      onPrimary: AppColors.onPrimary,
      primaryContainer: AppColors.primaryContainer,
      onPrimaryContainer: AppColors.onPrimaryContainer,
      secondary: AppColors.secondary,
      onSecondary: AppColors.onSecondary,
      secondaryContainer: AppColors.secondaryContainer,
      onSecondaryContainer: AppColors.onSecondaryContainer,
      tertiary: AppColors.tertiary,
      onTertiary: AppColors.onTertiary,
      tertiaryContainer: AppColors.tertiaryContainer,
      onTertiaryContainer: AppColors.onTertiaryContainer,
      error: AppColors.error,
      onError: AppColors.onError,
      errorContainer: AppColors.errorContainer,
      onErrorContainer: AppColors.onErrorContainer,
      background: AppColors.background,
      onBackground: AppColors.textPrimary,
      surface: AppColors.surface,
      onSurface: AppColors.textPrimary,
      surfaceVariant: AppColors.surfaceVariant,
      onSurfaceVariant: AppColors.textSecondary,
      outline: AppColors.outline,
      outlineVariant: AppColors.outlineVariant,
      shadow: AppColors.shadow,
      scrim: AppColors.scrim,
      inverseSurface: AppColors.inverseSurface,
      onInverseSurface: AppColors.onInverseSurface,
      inversePrimary: AppColors.primaryDark,
      surfaceTint: AppColors.primary,
    );

final ThemeData _lightTheme = ThemeData(
  useMaterial3: true,
  brightness: Brightness.light,
  colorScheme: _lightColorScheme,
  scaffoldBackgroundColor: AppColors.background,
  primaryColor: AppColors.primary,
  splashColor: AppColors.primaryHover,
  highlightColor: Colors.transparent,
  hoverColor: AppColors.primaryHover,
  disabledColor: AppColors.disabled,
  fontFamily: GoogleFonts.manrope().fontFamily,
  textTheme: AppTypography.lightTheme,
  iconTheme: const IconThemeData(color: AppColors.iconPrimary),
  appBarTheme: const AppBarTheme(
    backgroundColor: AppColors.surface,
    foregroundColor: AppColors.textPrimary,
    surfaceTintColor: Colors.transparent,
    elevation: 0,
  ),
);
