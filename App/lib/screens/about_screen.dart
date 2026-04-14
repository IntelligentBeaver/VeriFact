import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:verifact_app/extensions/context_extensions.dart';
import 'package:verifact_app/utils/constants/sizes.dart';

class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('About Us', style: context.text.headlineSmall),
        backgroundColor: Colors.transparent,
        elevation: 0,
        foregroundColor: context.color.onSurface,
      ),
      backgroundColor: context.theme.scaffoldBackgroundColor,
      body: Padding(
        padding: EdgeInsets.all(AppSizes.md),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('VeriFact', style: context.text.displaySmall),
            SizedBox(height: AppSizes.md),
            Text(
              'VeriFact helps you verify claims and search documents quickly.\n\nFor more information check out our GitHub repository.',
              style: context.text.bodyMedium,
            ),
            SizedBox(height: AppSizes.sm),
            Row(
              children: [
                // Small GitHub icon button
                IconButton(
                  padding: EdgeInsets.zero,
                  constraints: const BoxConstraints(),
                  icon: Icon(
                    Icons.code,
                    size: 18,
                    color: context.color.primary,
                  ),
                  tooltip: 'Open GitHub',
                  onPressed: () async {
                    final uri = Uri.parse(
                      'https://github.com/IntelligentBeaver/VeriFact',
                    );
                    try {
                      await launchUrl(
                        uri,
                        mode: LaunchMode.externalApplication,
                      );
                    } catch (_) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(content: Text('Could not open GitHub')),
                      );
                    }
                  },
                ),
                SizedBox(width: AppSizes.xsSm),
                Text(
                  'Open repository',
                  style: context.text.bodyMedium?.copyWith(
                    color: context.color.primary,
                  ),
                ),
              ],
            ),
            SizedBox(height: AppSizes.md),
            const Divider(),
            SizedBox(height: AppSizes.md),
            Text('Credits', style: context.text.headlineSmall),
            SizedBox(height: AppSizes.sm),
            // Credit entries
            Text('Aman Sheikh', style: context.text.bodyLarge),
            const SizedBox(height: 4),
            Text(
              'abameikh@gmail.com',
              style: context.text.bodyMedium?.copyWith(
                color: context.color.primary,
              ),
            ),
            SizedBox(height: AppSizes.sm),
            Text('Shreya Khannal', style: context.text.bodyLarge),
            const SizedBox(height: 4),
            Text(
              'shreya.211546@ncit.edu.np',
              style: context.text.bodyMedium?.copyWith(
                color: context.color.primary,
              ),
            ),
            SizedBox(height: AppSizes.sm),
            Text('Shikshya K.C.', style: context.text.bodyLarge),
            const SizedBox(height: 4),
            Text(
              'shikshyachettri@gmail.com',
              style: context.text.bodyMedium?.copyWith(
                color: context.color.primary,
              ),
            ),
            SizedBox(height: AppSizes.sm),
            Text('Prashant Chhetrii', style: context.text.bodyLarge),
            const SizedBox(height: 4),
            Text(
              'prashantchhetrii465@gmail.com',
              style: context.text.bodyMedium?.copyWith(
                color: context.color.primary,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
