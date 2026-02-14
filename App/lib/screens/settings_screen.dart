import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:verifact_app/extensions/context_extensions.dart';
import 'package:verifact_app/flavors/flavor_config.dart';
import 'package:verifact_app/screens/about_screen.dart';
import 'package:verifact_app/utils/constants/sizes.dart';
import 'package:verifact_app/utils/helpers/helper_functions.dart';
import 'package:verifact_app/utils/notifiers/theme_notifier.dart';

class SettingsScreen extends ConsumerStatefulWidget {
  const SettingsScreen({super.key});

  @override
  ConsumerState<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends ConsumerState<SettingsScreen> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: context.theme.scaffoldBackgroundColor,
      appBar: AppBar(
        elevation: 0,
        backgroundColor: Colors.transparent,
        foregroundColor: context.color.onSurface,
        title: Text(
          'Settings',
          style: context.text.headlineLarge?.copyWith(
            fontWeight: FontWeight.w900,
          ),
        ),
      ),
      body: Padding(
        padding: EdgeInsets.all(AppSizes.md),
        child: Column(
          children: [
            Container(
              width: double.infinity,
              padding: EdgeInsets.symmetric(
                horizontal: AppSizes.md,
                vertical: AppSizes.smMd,
              ),
              decoration: BoxDecoration(
                color: context.color.surface,
                borderRadius: BorderRadius.circular(AppSizes.borderRadiusLg),
                border: Border.all(color: context.color.outlineVariant),
              ),
              child: Column(
                children: [
                  // Theme row
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Row(
                        children: [
                          Icon(
                            Icons.brightness_6_outlined,
                            color: context.color.primary,
                          ),
                          SizedBox(width: AppSizes.smMd),
                          Text('Theme', style: context.text.bodyLarge),
                        ],
                      ),
                      ConstrainedBox(
                        constraints: BoxConstraints(
                          maxHeight: AppSizes.xl + 4.h,
                        ),
                        child: FittedBox(
                          fit: BoxFit.fitWidth,
                          child: Consumer(
                            builder: (context, ref, _) {
                              final themeState = ref.watch(themeProvider);
                              final isDark =
                                  (themeState.value ?? ThemeMode.light) ==
                                  ThemeMode.dark;
                              return Switch(
                                value: isDark,
                                onChanged: (v) async {
                                  await ref
                                      .read(themeProvider.notifier)
                                      .toggleTheme();
                                },
                                activeThumbColor: context.color.primary,
                                inactiveThumbColor:
                                    context.color.onSurfaceVariant,
                              );
                            },
                          ),
                        ),
                      ),
                    ],
                  ),

                  Divider(color: context.color.outlineVariant),

                  // About row
                  InkWell(
                    onTap: () => Navigator.of(context).push(
                      MaterialPageRoute<void>(
                        builder: (_) => const AboutScreen(),
                      ),
                    ),
                    child: Padding(
                      padding: EdgeInsets.symmetric(vertical: AppSizes.smMd),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Row(
                            children: [
                              Icon(
                                Icons.info_outline,
                                color: context.color.primary,
                              ),
                              SizedBox(width: AppSizes.smMd),
                              Text('About Us', style: context.text.bodyLarge),
                            ],
                          ),
                          Icon(
                            Icons.chevron_right,
                            color: context.color.onSurfaceVariant,
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),

            SizedBox(height: AppSizes.lg),

            // Change baseURL (long press to edit)
            SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                style: ElevatedButton.styleFrom(
                  backgroundColor: context.color.errorContainer,
                  foregroundColor: context.color.onErrorContainer,
                  padding: EdgeInsets.symmetric(vertical: AppSizes.smMd),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(
                      AppSizes.borderRadiusLg,
                    ),
                  ),
                  elevation: 0,
                ),
                onPressed: () {
                  showInfoSnackbar('Long-press to change base URL');
                },
                onLongPress: () async {
                  final prefs = await SharedPreferences.getInstance();
                  final current =
                      prefs.getString('override_base_url') ??
                      FlavorConfig.instance.baseUrl;
                  final controller = TextEditingController(text: current);
                  final result = await showDialog<String?>(
                    context: context,
                    builder: (ctx) => AlertDialog(
                      title: Text(
                        'Change base URL',
                        style: context.text.titleLarge,
                      ),
                      content: TextField(
                        controller: controller,
                        decoration: const InputDecoration(
                          hintText: 'https://your-ngrok-url.ngrok.io',
                        ),
                        keyboardType: TextInputType.url,
                      ),
                      actions: [
                        TextButton(
                          onPressed: () => Navigator.of(ctx).pop(),
                          child: const Text('Cancel'),
                        ),
                        TextButton(
                          onPressed: () =>
                              Navigator.of(ctx).pop(controller.text.trim()),
                          child: const Text('OK'),
                        ),
                      ],
                    ),
                  );

                  if (result == null) return;
                  final trimmed = result.trim();
                  if (trimmed.isEmpty) {
                    await prefs.remove('override_base_url');
                    showSuccessSnackbar('Base URL override removed');
                  } else {
                    await prefs.setString('override_base_url', trimmed);
                    showSuccessSnackbar('Base URL updated');
                  }
                },
                icon: const Icon(Icons.link),
                label: Text('Change baseURL', style: context.text.labelLarge),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
