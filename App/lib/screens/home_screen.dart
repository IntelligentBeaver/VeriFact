import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:lucide_icons_flutter/lucide_icons.dart';
import 'package:skeletonizer/skeletonizer.dart';
import 'package:verifact_app/extensions/context_extensions.dart';
import 'package:verifact_app/models/qa_model.dart';
import 'package:verifact_app/models/retriever_model.dart';
import 'package:verifact_app/screens/verifier_result_screen.dart';
import 'package:verifact_app/utils/constants/sizes.dart';
import 'package:verifact_app/utils/helpers/helper_functions.dart';
import 'package:verifact_app/utils/notifiers/home_search_notifier.dart';
import 'package:verifact_app/utils/notifiers/qa_notifier.dart';
import 'package:verifact_app/utils/notifiers/retriever_notifier.dart';
import 'package:verifact_app/widgets/results/custom_doc_result_card_skeleton.dart';
import 'package:verifact_app/widgets/results/custom_doc_search_result_card.dart';
import 'package:verifact_app/widgets/results/custom_qa_result_card.dart';
import 'package:verifact_app/widgets/results/custom_qa_result_card_skeleton.dart';
import 'package:verifact_app/widgets/results/custom_results_header.dart';
import 'package:verifact_app/widgets/results/custom_source_card.dart';

class HomeScreen extends ConsumerStatefulWidget {
  const HomeScreen({super.key});

  @override
  ConsumerState<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends ConsumerState<HomeScreen> {
  final _searchController = TextEditingController();
  final FocusNode _searchFocus = FocusNode();
  int _currentIndex = 0;
  bool _hasSubmitted = false;
  HomeSearchMode _lastSubmittedMode = HomeSearchMode.verifier;

  @override
  void dispose() {
    _searchController.dispose();
    _searchFocus.dispose();
    super.dispose();
  }

  Future<void> _submitSearch() async {
    final query = _searchController.text.trim();
    if (query.isEmpty) return;

    var mode = ref.read(homeSearchProvider);
    // If verifier mode (default) — navigate to verifier results page
    if (mode == HomeSearchMode.verifier) {
      Navigator.of(context).push(
        MaterialPageRoute<void>(
          builder: (_) => VerifierResultScreen(claim: query),
        ),
      );
      return;
    }

    // For QA/Doc modes, perform the usual fetch and show results inline
    setState(() {
      _hasSubmitted = true;
      _lastSubmittedMode = mode;
    });

    switch (mode) {
      case HomeSearchMode.qa:
        await ref.read(qaProvider.notifier).fetchAnswer(query);
        break;
      case HomeSearchMode.doc:
        await ref.read(retrieverProvider.notifier).search(query);
        break;
      default:
        break;
    }
  }

  void _resetSearch() {
    setState(() {
      _hasSubmitted = false;
      _lastSubmittedMode = HomeSearchMode.verifier;
    });
    _searchController.clear();
    ref.read(homeSearchProvider.notifier).selectMode(HomeSearchMode.verifier);
  }

  Future<void> _showQuickMenu(BuildContext context) async {
    await showModalBottomSheet<void>(
      context: context,
      backgroundColor: context.color.surface,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(
          top: Radius.circular(AppSizes.borderRadiusLg),
        ),
      ),
      builder: (context) => const QuickMenuSheet(),
    );
  }

  String _titleForMode(HomeSearchMode mode) {
    switch (mode) {
      case HomeSearchMode.qa:
        return 'QA System';
      case HomeSearchMode.doc:
        return 'Doc Search';
      case HomeSearchMode.verifier:
        return 'Good morning, Alex';
    }
  }

  String _subtitleForMode(HomeSearchMode mode) {
    switch (mode) {
      case HomeSearchMode.qa:
        return 'How can I help you find the answers of today?';
      case HomeSearchMode.doc:
        return 'What document do you need help searching?';
      case HomeSearchMode.verifier:
        return 'How can I help you fact-check today?';
    }
  }

  @override
  Widget build(BuildContext context) {
    final mode = ref.watch(homeSearchProvider);
    final qaState = ref.watch(qaProvider);
    final retrieverState = ref.watch(retrieverProvider);

    return GestureDetector(
      behavior: HitTestBehavior.translucent,
      onTap: () => FocusScope.of(context).unfocus(),
      child: Scaffold(
        backgroundColor: context.color.background,
        bottomNavigationBar: HomeBottomNav(
          currentIndex: _currentIndex,
          onTap: (index) => setState(() => _currentIndex = index),
        ),
        body: SafeArea(
          child: Padding(
            padding: EdgeInsets.symmetric(
              horizontal: AppSizes.smMd,
              vertical: AppSizes.md,
            ),
            child: Column(
              children: [
                Expanded(
                  child: AnimatedSwitcher(
                    duration: const Duration(milliseconds: 300),
                    transitionBuilder: (child, animation) => FadeTransition(
                      opacity: animation,
                      child: FadeTransition(
                        opacity: Tween<double>(
                          begin: 0.5,
                          end: 1,
                        ).animate(animation),
                        child: child,
                      ),
                    ),
                    child: _hasSubmitted
                        ? _ResultsSection(
                            key: const ValueKey('results'),
                            hasSubmitted: _hasSubmitted,
                            mode: _lastSubmittedMode,
                            qaState: qaState,
                            retrieverState: retrieverState,
                          )
                        : Column(
                            key: const ValueKey('header'),
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              _HomeHeader(
                                title: _titleForMode(mode),
                                subtitle: _subtitleForMode(mode),
                              ),
                              SizedBox(height: AppSizes.lg),
                              _QuickActionsSection(
                                onModeSelected: (selected) => ref
                                    .read(homeSearchProvider.notifier)
                                    .selectMode(selected),
                                onTap: (label) {
                                  showInfoSnackbar('$label tapped');
                                },
                              ),
                            ],
                          ),
                  ),
                ),
                _SearchBarContainer(
                  mode: mode,
                  controller: _searchController,
                  focusNode: _searchFocus,
                  onModeCleared: _resetSearch,
                  onPlusTap: () => _showQuickMenu(context),
                  onSubmitted: (_) => _submitSearch(),
                ),
                SizedBox(height: AppSizes.smMd),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _HomeHeader extends StatelessWidget {
  const _HomeHeader({required this.title, required this.subtitle});

  final String title;
  final String subtitle;

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Text(
          title,
          textAlign: TextAlign.center,
          style: context.text.displayLarge?.copyWith(
            fontWeight: FontWeight.w900,
          ),
        ),
        SizedBox(height: AppSizes.xsSm),
        Text(
          subtitle,
          textAlign: TextAlign.center,
          style: context.text.bodySmall,
        ),
      ],
    );
  }
}

class _QuickActionsSection extends StatelessWidget {
  const _QuickActionsSection({
    required this.onTap,
    required this.onModeSelected,
  });

  final void Function(String label) onTap;
  final void Function(HomeSearchMode mode) onModeSelected;

  @override
  Widget build(BuildContext context) {
    final actions = [
      const _QuickActionData('QA System', LucideIcons.messagesSquare),
      const _QuickActionData('Doc Search', LucideIcons.fileSearch),
      const _QuickActionData('Scan Image', LucideIcons.camera),
      const _QuickActionData('Upload Image', LucideIcons.fileUp),
    ];

    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Row(
          children: [
            Expanded(
              child: _QuickActionButton(
                data: actions[0],
                onTap: () => onModeSelected(HomeSearchMode.qa),
              ),
            ),
            SizedBox(width: AppSizes.smMd),
            Expanded(
              child: _QuickActionButton(
                data: actions[1],
                onTap: () => onModeSelected(HomeSearchMode.doc),
              ),
            ),
          ],
        ),
        SizedBox(height: AppSizes.smMd),
        Row(
          children: [
            Expanded(
              child: _QuickActionButton(
                data: actions[2],
                onTap: () => onTap(actions[2].label),
              ),
            ),
            SizedBox(width: AppSizes.smMd),
            Expanded(
              child: _QuickActionButton(
                data: actions[3],
                onTap: () => onTap(actions[3].label),
              ),
            ),
          ],
        ),
      ],
    );
  }
}

class _QuickActionButton extends StatelessWidget {
  const _QuickActionButton({
    required this.data,
    required this.onTap,
  });

  final _QuickActionData data;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      borderRadius: BorderRadius.circular(AppSizes.borderRadiusXl),
      onTap: onTap,
      child: Container(
        padding: EdgeInsets.symmetric(
          horizontal: AppSizes.md,
          vertical: AppSizes.smMd,
        ),
        decoration: BoxDecoration(
          color: context.color.surface,
          borderRadius: BorderRadius.circular(AppSizes.borderRadiusXl),
          border: Border.all(color: context.color.outlineVariant),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              data.icon,
              size: AppSizes.iconSm,
              color: data.label == 'Doc Search' || data.label == 'Scan Image'
                  ? context.color.secondary
                  : context.color.primary,
            ),
            SizedBox(width: AppSizes.sm),
            Flexible(
              child: Text(
                data.label,
                overflow: TextOverflow.ellipsis,
                style: context.text.labelLarge?.copyWith(),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _SearchBarContainer extends ConsumerStatefulWidget {
  const _SearchBarContainer({
    required this.mode,
    required this.controller,
    required this.focusNode,
    required this.onModeCleared,
    required this.onPlusTap,
    required this.onSubmitted,
    super.key,
  });

  final HomeSearchMode mode;
  final TextEditingController controller;
  final FocusNode focusNode;
  final VoidCallback onModeCleared;
  final VoidCallback onPlusTap;
  final ValueChanged<String> onSubmitted;

  @override
  ConsumerState<_SearchBarContainer> createState() =>
      _SearchBarContainerState();
}

class _SearchBarContainerState extends ConsumerState<_SearchBarContainer> {
  @override
  void initState() {
    super.initState();
    widget.focusNode.addListener(_onFocusChange);
  }

  @override
  void dispose() {
    widget.focusNode.removeListener(_onFocusChange);
    super.dispose();
  }

  void _onFocusChange() => setState(() {});

  @override
  Widget build(BuildContext context) {
    final mode = widget.mode;
    final showChip = mode != HomeSearchMode.verifier;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (showChip) ...[
          SizedBox(height: AppSizes.smMd),
          _ModeChip(mode: mode, onClear: widget.onModeCleared),
        ],
        SizedBox(height: AppSizes.smMd),
        Container(
          padding: EdgeInsets.fromLTRB(
            AppSizes.smMd,
            AppSizes.xs,
            AppSizes.xs,
            AppSizes.xs,
          ),
          decoration: BoxDecoration(
            color: context.color.surface,
            borderRadius: BorderRadius.circular(AppSizes.borderRadiusXl + 8.r),
            border: Border.all(
              color: widget.focusNode.hasFocus
                  ? context.color.primary
                  : context.color.outlineVariant,
            ),
          ),
          child: Row(
            children: [
              Icon(
                LucideIcons.search,
                size: AppSizes.iconSm,
                color: context.color.onSurfaceVariant,
              ),
              SizedBox(width: AppSizes.sm),
              Expanded(
                child: TextField(
                  focusNode: widget.focusNode,
                  controller: widget.controller,
                  onSubmitted: widget.onSubmitted,
                  textInputAction: TextInputAction.search,
                  style: context.text.bodyMedium?.copyWith(
                    color: context.color.onSurface,
                  ),
                  decoration: InputDecoration(
                    focusColor: context.color.primary,
                    hoverColor: context.color.primary,
                    hintText: mode == HomeSearchMode.verifier
                        ? 'Verify claim...'
                        : mode == HomeSearchMode.qa
                        ? 'Ask a question...'
                        : 'Search documents...',
                    hintStyle: context.text.bodySmall?.copyWith(
                      color: context.color.onSurfaceVariant,
                    ),
                    isDense: true,
                    border: InputBorder.none,
                  ),
                ),
              ),
              SizedBox(width: AppSizes.smMd),
              InkWell(
                onTap: widget.onPlusTap,
                borderRadius: BorderRadius.circular(
                  AppSizes.borderRadiusXl + 8.r,
                ),
                child: Container(
                  width: AppSizes.iconLg,
                  height: AppSizes.iconLg,
                  // decoration: BoxDecoration(
                  //   color: context.color.surfaceContainer,
                  //   borderRadius: BorderRadius.circular(
                  //     AppSizes.borderRadiusXl,
                  //   ),
                  // ),
                  child: Icon(
                    LucideIcons.plus,
                    size: AppSizes.iconMd,
                    color: context.color.onSurfaceVariant,
                  ),
                ),
              ),
              SizedBox(width: AppSizes.smMd),
              // Submit (up arrow) button — enabled when there is text
              ValueListenableBuilder<TextEditingValue>(
                valueListenable: widget.controller,
                builder: (context, value, _) {
                  final enabled = value.text.trim().isNotEmpty;
                  return InkWell(
                    onTap: enabled
                        ? () =>
                              widget.onSubmitted(widget.controller.text.trim())
                        : null,
                    child: Container(
                      width: AppSizes.iconLg,
                      height: AppSizes.iconLg,
                      decoration: BoxDecoration(
                        color: enabled
                            ? context.color.primary
                            : context.color.surfaceContainer,
                        borderRadius: BorderRadius.circular(
                          AppSizes.borderRadiusXl,
                        ),
                      ),
                      child: Icon(
                        LucideIcons.arrowUp,
                        size: AppSizes.iconSm,
                        color: enabled
                            ? context.color.onPrimary
                            : context.color.onSurfaceVariant,
                      ),
                    ),
                  );
                },
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _ModeChip extends StatelessWidget {
  const _ModeChip({
    required this.mode,
    required this.onClear,
  });

  final HomeSearchMode mode;
  final VoidCallback onClear;

  String _modeLabel() {
    switch (mode) {
      case HomeSearchMode.qa:
        return 'QA System';
      case HomeSearchMode.doc:
        return 'Doc Search';
      case HomeSearchMode.verifier:
        return 'Verifier';
    }
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onClear,
      child: Container(
        padding: EdgeInsets.symmetric(
          horizontal: AppSizes.md,
          vertical: AppSizes.xsSm,
        ),
        decoration: BoxDecoration(
          color: context.color.primaryContainer,
          borderRadius: BorderRadius.circular(AppSizes.borderRadiusXl),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              _modeLabel(),
              style: context.text.labelSmall?.copyWith(
                color: context.color.onPrimaryContainer,
                fontWeight: FontWeight.w600,
              ),
            ),
            SizedBox(width: AppSizes.xsSm),
            Icon(
              LucideIcons.x,
              size: AppSizes.iconSm,
              color: context.color.onPrimaryContainer,
            ),
          ],
        ),
      ),
    );
  }
}

class ChatInputBar extends StatelessWidget {
  const ChatInputBar({
    required this.controller,
    required this.hintText,
    required this.onPlusTap,
    required this.onSubmitted,
    super.key,
  });

  final TextEditingController controller;
  final String hintText;
  final VoidCallback onPlusTap;
  final ValueChanged<String> onSubmitted;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.symmetric(
        horizontal: AppSizes.md,
        vertical: AppSizes.smMd,
      ),
      decoration: BoxDecoration(
        color: context.color.surface,
        borderRadius: BorderRadius.circular(AppSizes.borderRadiusLg),
        border: Border.all(color: context.color.outlineVariant),
      ),
      child: Row(
        children: [
          Icon(
            LucideIcons.search,
            size: AppSizes.iconSm,
            color: context.color.onSurfaceVariant,
          ),
          SizedBox(width: AppSizes.smMd),
          Expanded(
            child: TextField(
              controller: controller,
              onSubmitted: onSubmitted,
              textInputAction: TextInputAction.search,
              style: context.text.bodyMedium?.copyWith(
                color: context.color.onSurface,
              ),
              decoration: InputDecoration(
                hintText: hintText,
                hintStyle: context.text.bodySmall?.copyWith(
                  color: context.color.onSurfaceVariant,
                ),
                isDense: true,
                border: InputBorder.none,
              ),
            ),
          ),
          SizedBox(width: AppSizes.smMd),
          InkWell(
            onTap: onPlusTap,
            borderRadius: BorderRadius.circular(AppSizes.borderRadiusMd),
            child: Container(
              width: AppSizes.iconLg,
              height: AppSizes.iconLg,
              decoration: BoxDecoration(
                color: context.color.surfaceVariant,
                borderRadius: BorderRadius.circular(AppSizes.borderRadiusMd),
              ),
              child: Icon(
                LucideIcons.plus,
                size: AppSizes.iconSm,
                color: context.color.onSurfaceVariant,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

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
        BottomNavigationBarItem(
          icon: Icon(LucideIcons.compass),
          label: 'Explore',
        ),
        BottomNavigationBarItem(
          icon: Icon(LucideIcons.user),
          label: 'Profile',
        ),
      ],
    );
  }
}

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
            icon: LucideIcons.messagesSquare,
            label: 'QA System',
          ),
          const _MenuTile(
            icon: LucideIcons.fileSearch,
            label: 'Doc Search',
          ),
          const _MenuTile(
            icon: LucideIcons.scan,
            label: 'Scan Image',
          ),
          const _MenuTile(
            icon: LucideIcons.upload,
            label: 'Upload Image',
          ),
        ],
      ),
    );
  }
}

class _MenuTile extends StatelessWidget {
  const _MenuTile({required this.icon, required this.label});

  final IconData icon;
  final String label;

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
        child: Icon(icon, color: context.color.primary),
      ),
      title: Text(
        label,
        style: context.text.labelMedium?.copyWith(
          fontWeight: FontWeight.w600,
          color: context.color.onSurface,
        ),
      ),
      onTap: () => Navigator.pop(context),
    );
  }
}

class _ResultsSection extends StatelessWidget {
  const _ResultsSection({
    required this.hasSubmitted,
    required this.mode,
    required this.qaState,
    required this.retrieverState,
    super.key,
  });

  final bool hasSubmitted;
  final HomeSearchMode mode;
  final AsyncValue<QAModel?> qaState;
  final AsyncValue<RetrieverResponse?> retrieverState;

  @override
  Widget build(BuildContext context) {
    switch (mode) {
      case HomeSearchMode.qa:
        return _qaResults(context, qaState);
      case HomeSearchMode.doc:
        return _retrieverResults(context, retrieverState);
      case HomeSearchMode.verifier:
        return const SizedBox.shrink();
    }
  }

  Widget _qaResults(BuildContext context, AsyncValue<QAModel?> state) {
    return state.when(
      loading: () => Skeletonizer(
        enabled: true,
        child: _qaSkeletonLoader(context),
      ),
      error: (err, _) => _errorCard(context, err.toString()),
      data: (data) {
        if (data == null) return _emptyCard(context, 'No answers yet.');

        final (conclusion, evidence) = _parseAnswer(data.answer);

        return ListView(
          padding: EdgeInsets.zero,
          children: [
            Text(
              'Your Answer',
              style: context.text.displayMedium?.copyWith(
                fontWeight: FontWeight.w900,
                color: context.color.onSurface,
              ),
            ),
            SizedBox(height: AppSizes.mdLg),
            CustomQAResultCard(
              conclusion: conclusion,
              evidence: evidence,
            ),
            if (data.sources.isNotEmpty) ...[
              SizedBox(height: AppSizes.md),
              Row(
                spacing: AppSizes.xsSm,
                children: [
                  Icon(
                    LucideIcons.bookSearch500,
                    size: 18.r,
                    color: context.color.primary,
                  ),
                  Text(
                    'Sources',
                    style: context.text.headlineLarge?.copyWith(
                      fontWeight: FontWeight.w900,
                      color: context.color.onSurface,
                    ),
                  ),
                ],
              ),
              SizedBox(height: AppSizes.md),
              ...data.sources.map(
                (source) => Padding(
                  padding: EdgeInsets.only(bottom: AppSizes.smMd),
                  child: CustomSourceCard(
                    title: source.title.isEmpty ? 'Source' : source.title,
                    text: source.text,
                    url: source.url,
                    score: source.score,
                  ),
                ),
              ),
            ],
          ],
        );
      },
    );
  }

  Widget _qaSkeletonLoader(BuildContext context) {
    return ListView(
      padding: EdgeInsets.zero,
      children: [
        Container(
          height: 32.0,
          width: 200.0,
          color: context.color.outlineVariant,
        ),
        SizedBox(height: AppSizes.mdLg),
        const CustomQAResultCardSkeleton(),
        SizedBox(height: AppSizes.lg),
        Container(
          height: 20.0,
          width: 100.0,
          color: context.color.outlineVariant,
        ),
        SizedBox(height: AppSizes.mdLg),
        ...List.generate(
          3,
          (_) => Padding(
            padding: EdgeInsets.only(bottom: AppSizes.smMd),
            child: const CustomDocResultCardSkeleton(),
          ),
        ),
      ],
    );
  }

  Widget _retrieverResults(
    BuildContext context,
    AsyncValue<RetrieverResponse?> state,
  ) {
    return state.when(
      loading: () => Skeletonizer(
        child: _docSkeletonLoader(context),
      ),
      error: (err, _) => _errorCard(context, err.toString()),
      data: (data) {
        if (data == null || data.results.isEmpty) {
          return _emptyCard(context, 'No documents yet.');
        }

        return Column(
          children: [
            const CustomResultsHeader(title: 'Your Results'),
            SizedBox(height: AppSizes.mdLg),
            Expanded(
              child: ListView.separated(
                padding: EdgeInsets.zero,
                itemCount: data.results.length,
                separatorBuilder: (_, _) => SizedBox(height: AppSizes.smMd),
                itemBuilder: (context, index) {
                  final result = data.results[index];
                  final passage = result.passage;
                  final author = result.passage?.author ?? 'Unknown Author';
                  final url = result.passage?.url;
                  final title =
                      passage?.title ?? passage?.sectionHeading ?? 'Result';
                  final snippet = (passage?.text ?? '').replaceAll('\n', ' ');
                  final score =
                      result.finalScore ??
                      result.crossScore ??
                      result.faissScore;

                  return CustomDocSearchResultCard(
                    title: title,
                    content: snippet.isEmpty
                        ? 'No preview available.'
                        : snippet.length > 320
                        ? '${snippet.substring(0, 320)}...'
                        : snippet,
                    author: author,
                    score: score,
                    url: url,
                  );
                },
              ),
            ),
          ],
        );
      },
    );
  }

  Widget _docSkeletonLoader(BuildContext context) {
    return Column(
      children: [
        const CustomResultsHeader(title: 'Your Results'),
        SizedBox(height: AppSizes.mdLg),
        Expanded(
          child: ListView.separated(
            padding: EdgeInsets.zero,
            itemCount: 6,
            separatorBuilder: (_, __) => SizedBox(height: AppSizes.smMd),
            itemBuilder: (_, __) => CustomDocResultCardSkeleton(),
          ),
        ),
      ],
    );
  }

  (String conclusion, String evidence) _parseAnswer(String answer) {
    var conclusion = '';
    var evidence = '';

    // Try to split by "Conclusion:" and "Evidence:" labels
    final conclusionMatch = RegExp(
      r'Conclusion:\s*(.+?)(?=\n\nEvidence:|Evidence:|$)',
      dotAll: true,
    ).firstMatch(answer);
    final evidenceMatch = RegExp(
      r'Evidence:\s*(.+?)$',
      dotAll: true,
    ).firstMatch(answer);

    if (conclusionMatch != null) {
      conclusion = conclusionMatch.group(1)?.trim() ?? '';
    }
    if (evidenceMatch != null) {
      evidence = evidenceMatch.group(1)?.trim() ?? '';
    }

    // If parsing failed, use the entire text as conclusion
    if (conclusion.isEmpty && evidence.isEmpty) {
      conclusion = answer;
    }

    return (conclusion, evidence);
  }

  Widget _errorCard(BuildContext context, String message) {
    return Card(
      margin: EdgeInsets.zero,
      color: context.color.errorContainer,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(AppSizes.borderRadiusLg),
      ),
      child: Padding(
        padding: EdgeInsets.all(AppSizes.md),
        child: Text(
          message,
          style: context.text.bodySmall?.copyWith(
            color: context.color.onErrorContainer,
          ),
        ),
      ),
    );
  }

  Widget _emptyCard(BuildContext context, String message) {
    return Center(
      child: Text(
        message,
        style: context.text.bodySmall?.copyWith(
          color: context.color.onSurfaceVariant,
        ),
      ),
    );
  }
}

class _QuickActionData {
  const _QuickActionData(this.label, this.icon);

  final String label;
  final IconData icon;
}
