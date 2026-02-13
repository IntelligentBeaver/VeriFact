import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:verifact_app/utils/constants/colors.dart';
import 'package:verifact_app/utils/constants/sizes.dart';

class AppTypography {
  AppTypography._();

  static TextTheme lightTheme = TextTheme(
    // Display Texts
    displayLarge: _style(
      AppSizes.displayLg,
      FontWeight.w600,
      AppColors.textPrimary,
    ),
    displayMedium: _style(
      AppSizes.displayMd,
      FontWeight.w600,
      AppColors.textPrimary,
    ),
    displaySmall: _style(
      AppSizes.displaySm,
      FontWeight.w600,
      AppColors.textPrimary,
    ),

    headlineLarge: _style(
      AppSizes.headline,
      FontWeight.w600,
      AppColors.textPrimary,
    ),
    headlineMedium: _style(
      AppSizes.headline,
      FontWeight.w500,
      AppColors.textPrimary,
    ),
    headlineSmall: _style(
      AppSizes.headline,
      FontWeight.w400,
      AppColors.textPrimary,
    ),

    // Title Texts
    titleLarge: _style(AppSizes.title, FontWeight.w600, AppColors.textPrimary),
    titleMedium: _style(AppSizes.title, FontWeight.w500, AppColors.textPrimary),
    titleSmall: _style(AppSizes.title, FontWeight.w400, AppColors.textPrimary),

    // Body Texts
    bodyLarge: _style(AppSizes.body, FontWeight.w600, AppColors.textPrimary),
    bodyMedium: _style(AppSizes.body, FontWeight.w500, AppColors.textPrimary),
    bodySmall: _style(AppSizes.body, FontWeight.w400, AppColors.textPrimary),

    // Label Texts
    labelLarge: _style(AppSizes.label, FontWeight.w600, AppColors.textPrimary),
    labelMedium: _style(AppSizes.label, FontWeight.w500, AppColors.textPrimary),
    labelSmall: _style(AppSizes.label, FontWeight.w400, AppColors.textPrimary),
  );

  static TextTheme darkTheme = TextTheme(
    displayLarge: _style(
      AppSizes.displayLg,
      FontWeight.w600,
      AppColors.textPrimaryDark,
    ),
    displayMedium: _style(
      AppSizes.displayMd,
      FontWeight.w600,
      AppColors.textPrimaryDark,
    ),
    displaySmall: _style(
      AppSizes.displaySm,
      FontWeight.w600,
      AppColors.textPrimaryDark,
    ),
    headlineLarge: _style(
      AppSizes.headline,
      FontWeight.w600,
      AppColors.textPrimaryDark,
    ),
    headlineMedium: _style(
      AppSizes.headline,
      FontWeight.w500,
      AppColors.textPrimaryDark,
    ),
    headlineSmall: _style(
      AppSizes.headline,
      FontWeight.w400,
      AppColors.textPrimaryDark,
    ),
    titleLarge: _style(
      AppSizes.title,
      FontWeight.w600,
      AppColors.textPrimaryDark,
    ),
    titleMedium: _style(
      AppSizes.title,
      FontWeight.w500,
      AppColors.textPrimaryDark,
    ),
    titleSmall: _style(
      AppSizes.title,
      FontWeight.w400,
      AppColors.textPrimaryDark,
    ),
    bodyLarge: _style(
      AppSizes.body,
      FontWeight.w600,
      AppColors.textPrimaryDark,
    ),
    bodyMedium: _style(
      AppSizes.body,
      FontWeight.w500,
      AppColors.textPrimaryDark,
    ),
    bodySmall: _style(
      AppSizes.body,
      FontWeight.w400,
      AppColors.textPrimaryDark,
    ),
    labelLarge: _style(
      AppSizes.label,
      FontWeight.w600,
      AppColors.textPrimaryDark,
    ),
    labelMedium: _style(
      AppSizes.label,
      FontWeight.w500,
      AppColors.textPrimaryDark,
    ),
    labelSmall: _style(
      AppSizes.label,
      FontWeight.w400,
      AppColors.textPrimaryDark,
    ),
  );

  static TextStyle _style(
    double fontSize,
    FontWeight fontWeight,
    Color color,
  ) {
    return GoogleFonts.manrope(
      fontSize: fontSize,
      fontWeight: fontWeight,
      letterSpacing: 0.1,
      color: color,
    );
  }
}
