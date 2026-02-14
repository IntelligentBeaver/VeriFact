import 'package:flutter/material.dart';
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
              'VeriFact helps you verify claims and search documents quickly.\n\nFor more information visit our website or contact support.',
              style: context.text.bodyMedium,
            ),
          ],
        ),
      ),
    );
  }
}
