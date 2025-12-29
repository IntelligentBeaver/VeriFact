// Self Explainatory
// ignore_for_file: public_member_api_docs

import 'package:flutter/material.dart';

class AppColors {
  AppColors._();

  //? Only Dark mode colors have been confirmed
  //* Primary Brand Colors
  static const Color primary = Color(0xFFFF6B6B); //?unconfirmed
  static const Color onPrimary = Color(0xFFFFFFFF); //?unconfirmed

  static const Color primaryDark = Color(0xFFFF6B6B);
  static const Color primaryDark10 = Color(0x1AFF6B6B);
  static const Color onPrimaryDark = Color(0xFFFFFFFF);

  static const Color primaryClicked = Color(0xFF2E6634);
  static const Color primaryDisabled = Color(0xFFF4F4F5);

  static const Color primaryClickedDark = Color(0xFF2E6634);
  static const Color primaryDisabledDark = Color(0xFFF4F4F5);

  //* Secondary Colors
  static const Color secondary = Color(0xFF4ECDC4); //?unconfirmed
  static const Color secondaryDisabled = Color(0xFF497C78);
  static const Color onSecondary = AppColors.textPrimary; //?unconfirmed

  static const Color secondaryDark = Color(0xFF4ECDC4);
  static const Color secondaryDisabledDark = Color(0xFF497C78);
  static const Color onSecondaryDark = AppColors.textPrimary; //?unconfirmed

  //* Tertiary Colors
  static const Color tertiary = Color(0xFF8B7419);
  static const Color onTertiary = AppColors.textPrimary; //?unconfirmed
  // Dark
  static const Color tertiaryDark = Color(0xFFFFE066);
  static const Color onTertiaryDark = AppColors.textPrimary; //?unconfirmed

  //* Info Button
  static const Color info = Color(0xFF1E99DC); //?unconfirmed
  static const Color infoDark = Color(0xFF239FE1);

  //* Error
  static const Color error = Color(0xFFE5484D);
  static const Color errorContainer = Color(0xFFF4D8DB);
  static const Color onErrorContainer = Color(0xFF5F2128);

  static const Color errorDark = Color(0xFFFE2F51);
  static const Color onErrorDark = Color(0xFF5F2128);

  //* Success
  static const Color success = Color(0xFFD8F4E1);
  static const Color onSuccess = Color(0xFF215F27);

  static const Color successDark = Color(0xFFD8F4E1);
  static const Color onSuccessDark = Color(0xFF215F27);

  //* Background / Surface
  //Light
  static const Color background = Color(0xFFFFFFFF);
  static const Color surface = Color(0xFFFFFFFF); //?unconfirmed
  static const Color primaryBackground = Color(0x4DFF6B6B);
  // Dark
  static const Color backgroundDark = Color(0xFF070D1A);
  static const Color surfaceDark = Color(0xFF0F1727);
  static const Color primaryBackgroundDark = Color(0x33FF5940);

  //* Tile / Hover / Selector
  static const Color tile = Color(0x14007E6E); // 8% opacity
  static const Color tabDividerColor = Color(0xFFEEEEEE);
  static const Color tileDark = Color(0x1480FFEE); // 8% opacity
  static const Color tabDividerColorDark = Color(0xFFEEEEEE);

  //* Icon Colors
  // Light
  static const Color icon = Color(0xFF212121);
  static const Color icon70 = Color(0xFF4A4A4A);
  // Dark
  static const Color iconDark = Color(0xFFFFFFFF);
  static const Color iconDark70 = Color(0xB2FFFFFF);

  //* Text Colors
  // Light
  static const Color textPrimary = Color(0xFF1E1E1E);
  static const Color textSecondary = Color(0xFF5A5A5A);
  // Dark
  static const Color textPrimaryDark = Color(0xFFEDEDED);
  static const Color textSecondaryDark = Color(0xFF96A1AF);

  //* Button Colors
  // Light
  static const Color buttonPrimary = AppColors.primary;
  static const Color buttonPrimaryDisabled = AppColors.primaryDisabled;
  // static const Color buttonPrimaryPressed;

  static const Color buttonSecondary = AppColors.secondary;
  static const Color buttonSecondaryDisabled = AppColors.secondaryDisabled;
  // static const Color buttoSecondaryPressed;

  static const Color buttonBackground = AppColors.surface;

  // Dark
  static const Color buttonPrimaryDark = AppColors.primaryDark;
  static const Color buttonPrimaryDisabledDark = AppColors.primaryDisabledDark;

  static const Color buttonSecondaryDark = AppColors.secondaryDark;
  static const Color buttonSecondaryDisabledDark =
      AppColors.secondaryDisabledDark;

  static const Color buttonBackgroundDark = AppColors.surfaceDark;

  //* Extra
  // Light
  static const Color stroke1 = Color(0xFFC9C8CC);
  static const Color stroke2 = Color(0xFF233146);

  // Dark
  static const Color strokeDark1 = Color(0xFF58565C);
  static const Color strokeDark2 = Color(0xFF233146);

  static const Color textField = Color(0xFFF5F5F5);
  static const Color textFieldDark = Color(0xFF1A1A1A);

  static const Color hint = Color(0xFF4A4A4A);
  static const Color hintDark = Color(0xB2FFFFFF);
}
