import 'package:flutter/material.dart';
import 'package:lucide_icons_flutter/lucide_icons.dart';
import 'package:verifact_app/extensions/context_extensions.dart';

class HomeBottomNav extends StatelessWidget {
  const HomeBottomNav({
    required this.currentIndex,
    required this.onTap,
    super.key,
  });

  final int currentIndex;
  final ValueChanged<int> onTap;

  @override
  Widget build(BuildContext context) {
    return BottomNavigationBar(
      currentIndex: currentIndex,
      onTap: onTap,
      type: BottomNavigationBarType.fixed,
      selectedItemColor: context.color.primary,
      unselectedItemColor: context.color.onSurfaceVariant,
      selectedLabelStyle: context.text.labelSmall?.copyWith(
        fontWeight: FontWeight.w600,
      ),
      unselectedLabelStyle: context.text.labelSmall,
      items: const [
        BottomNavigationBarItem(
          icon: Icon(LucideIcons.house),
          label: 'Home',
        ),
        BottomNavigationBarItem(
          icon: Icon(LucideIcons.history),
          label: 'History',
        ),
        // BottomNavigationBarItem(
        //   icon: Icon(LucideIcons.compass),
        //   label: 'Explore',
        // ),
        // BottomNavigationBarItem(
        //   icon: Icon(LucideIcons.user),
        //   label: 'Profile',
        // ),
      ],
    );
  }
}
