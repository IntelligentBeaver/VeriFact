import 'package:flutter/material.dart';
import 'package:lucide_icons_flutter/lucide_icons.dart';
import 'package:verifact_app/extensions/context_extensions.dart';
import 'package:verifact_app/utils/constants/sizes.dart';

class QuickMenuSheet extends StatelessWidget {
  const QuickMenuSheet({super.key});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.fromLTRB(
        AppSizes.lg,
        AppSizes.mdLg,
        AppSizes.lg,
        AppSizes.lg,
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Quick Actions',
            style: context.text.titleMedium?.copyWith(
              fontWeight: FontWeight.w700,
              color: context.color.onSurface,
            ),
          ),
          SizedBox(height: AppSizes.smMd),
          const _MenuTile(
            icon: LucideIcons.camera,
            label: 'Scan Image',
            description: 'Take a photo of a medical claim',
          ),
          const _MenuTile(
            icon: LucideIcons.upload,
            label: 'Upload Image',
            description: 'Pick from gallery to verify a claim',
          ),
        ],
      ),
    );
  }
}

class _MenuTile extends StatelessWidget {
  const _MenuTile({
    required this.description,
    required this.icon,
    required this.label,
  });

  final IconData icon;
  final String label;
  final String description;

  @override
  Widget build(BuildContext context) {
    return ListTile(
      contentPadding: EdgeInsets.zero,
      leading: Container(
        width: AppSizes.iconLg,
        height: AppSizes.iconLg,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(AppSizes.borderRadiusMd),
        ),
        child: Icon(icon, color: context.color.secondary),
      ),
      title: Text(
        label,
        style: context.text.labelMedium?.copyWith(
          fontWeight: FontWeight.w600,
          color: context.color.onSurface,
        ),
      ),
      subtitle: Text(
        description,
        style: context.text.labelSmall?.copyWith(
          color: context.color.onSurfaceVariant,
        ),
      ),
      onTap: () => Navigator.pop(context),
    );
  }
}
