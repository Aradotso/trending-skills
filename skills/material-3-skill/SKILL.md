```markdown
---
name: material-3-skill
description: Claude Code skill for implementing Material Design 3 (Material You) UI — 30+ components, design tokens, theming, responsive layout, and MD3 compliance auditing across web, Flutter, and Jetpack Compose.
triggers:
  - implement material design 3 component
  - build MD3 UI with correct tokens
  - generate material you theme from seed color
  - scaffold responsive material 3 app shell
  - audit MD3 compliance of my app
  - create material design navigation pattern
  - apply material 3 color roles and typography
  - check if my UI follows material design 3 spec
---

# Material Design 3 Skill for Claude Code

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A comprehensive skill for implementing Google's [Material Design 3](https://m3.material.io/) (Material You) UI system. Covers 30+ components, design tokens, theming, responsive layout, motion, accessibility, and an MD3 compliance audit across **web** (`@material/web`), **Flutter**, and **Jetpack Compose**.

---

## Installation

```bash
# Clone and copy into Claude Code skills directory
git clone https://github.com/hamen/material-3-skill.git
cp -r material-3-skill ~/.claude/skills/material-3

# Or symlink for easy updates
ln -s /path/to/material-3-skill ~/.claude/skills/material-3
```

---

## Commands

| Command | What it does |
|---|---|
| `/material-3 component <description>` | Generate an MD3-compliant component |
| `/material-3 theme <seed-color>` | Generate a complete MD3 theme from a seed color |
| `/material-3 scaffold <description>` | Scaffold a responsive MD3 app shell |
| `/material-3 audit <URL or file path>` | Audit MD3 compliance across 10 categories |

```bash
/material-3 component Create a login form with email and password fields
/material-3 theme Generate a theme from seed color #1A73E8
/material-3 scaffold Create a responsive app shell with navigation rail and drawer
/material-3 audit ./src/App.tsx
/material-3 audit https://myapp.example.com
```

---

## What the Skill Covers

### Reference Files

| File | Contents |
|---|---|
| `SKILL.md` | Philosophy, decision trees, token overview, component table, audit procedure |
| `references/color-system.md` | 29+ color roles, tonal palettes, dynamic color, baseline CSS scheme |
| `references/component-catalog.md` | 30+ components with elements, attributes, code examples |
| `references/theming-and-dynamic-color.md` | Theme generation, brand colors, dark mode, runtime switching |
| `references/typography-and-shape.md` | Type scale (30 styles), shape corner tokens, elevation, motion tokens |
| `references/navigation-patterns.md` | Nav component selection, responsive transitions, complete shell |
| `references/layout-and-responsive.md` | 5 breakpoints, 3 canonical layouts, CSS Grid implementation |

---

## Core Concepts

### Design Tokens

MD3 uses CSS custom properties for all tokens. Always use tokens — never hardcode hex values.

```css
/* Color tokens */
--md-sys-color-primary
--md-sys-color-on-primary
--md-sys-color-primary-container
--md-sys-color-on-primary-container
--md-sys-color-secondary
--md-sys-color-surface
--md-sys-color-on-surface
--md-sys-color-surface-variant
--md-sys-color-error
--md-sys-color-background

/* Typography tokens */
--md-sys-typescale-display-large-font-size        /* 57px */
--md-sys-typescale-headline-medium-font-size      /* 28px */
--md-sys-typescale-body-large-font-size           /* 16px */
--md-sys-typescale-label-small-font-size          /* 11px */

/* Shape tokens */
--md-sys-shape-corner-extra-small   /* 4px */
--md-sys-shape-corner-small         /* 8px */
--md-sys-shape-corner-medium        /* 12px */
--md-sys-shape-corner-large         /* 16px */
--md-sys-shape-corner-extra-large   /* 28px */
--md-sys-shape-corner-full          /* 50% */

/* Elevation tokens */
--md-sys-elevation-level0  /* 0dp */
--md-sys-elevation-level1  /* 1dp */
--md-sys-elevation-level2  /* 3dp */
--md-sys-elevation-level3  /* 6dp */
--md-sys-elevation-level4  /* 8dp */
--md-sys-elevation-level5  /* 12dp */
```

---

## Web Implementation (`@material/web`)

### Install

```bash
npm install @material/web
```

### Import and Use Components

```html
<!-- index.html -->
<script type="module" src="./main.js"></script>

<md-filled-button>Save</md-filled-button>
<md-outlined-text-field label="Email" type="email"></md-outlined-text-field>
<md-navigation-bar>
  <md-navigation-tab label="Home" active>
    <md-icon slot="active-icon">home</md-icon>
    <md-icon slot="inactive-icon">home</md-icon>
  </md-navigation-tab>
</md-navigation-bar>
```

```js
// main.js — tree-shake by importing only what you need
import '@material/web/button/filled-button.js';
import '@material/web/button/outlined-button.js';
import '@material/web/textfield/outlined-text-field.js';
import '@material/web/textfield/filled-text-field.js';
import '@material/web/checkbox/checkbox.js';
import '@material/web/radio/radio.js';
import '@material/web/select/filled-select.js';
import '@material/web/select/select-option.js';
import '@material/web/dialog/dialog.js';
import '@material/web/fab/fab.js';
import '@material/web/icon/icon.js';
import '@material/web/iconbutton/icon-button.js';
import '@material/web/chips/chip-set.js';
import '@material/web/chips/filter-chip.js';
import '@material/web/chips/assist-chip.js';
import '@material/web/list/list.js';
import '@material/web/list/list-item.js';
import '@material/web/menu/menu.js';
import '@material/web/menu/menu-item.js';
import '@material/web/progress/linear-progress.js';
import '@material/web/progress/circular-progress.js';
import '@material/web/slider/slider.js';
import '@material/web/switch/switch.js';
import '@material/web/tabs/tabs.js';
import '@material/web/tabs/primary-tab.js';
import '@material/web/navigationbar/navigation-bar.js';
import '@material/web/navigationbar/navigation-tab.js';
import '@material/web/navigationdrawer/navigation-drawer.js';
import '@material/web/divider/divider.js';
import '@material/web/elevation/elevation.js';
import '@material/web/focus/md-focus-ring.js';
import '@material/web/ripple/ripple.js';
```

### Complete Login Form Component

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet" />
  <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet" />
  <style>
    :root {
      --md-sys-color-primary: #1A73E8;
      --md-sys-color-on-primary: #ffffff;
      --md-sys-color-primary-container: #D3E3FD;
      --md-sys-color-on-primary-container: #001D35;
      --md-sys-color-surface: #FDFCFF;
      --md-sys-color-on-surface: #1A1C1E;
      font-family: 'Roboto', sans-serif;
    }
    .login-card {
      background: var(--md-sys-color-surface);
      border-radius: var(--md-sys-shape-corner-extra-large, 28px);
      padding: 32px;
      max-width: 400px;
      margin: 64px auto;
      box-shadow: var(--md-sys-elevation-level2, 0 1px 2px rgba(0,0,0,.3), 0 2px 6px 2px rgba(0,0,0,.15));
    }
    .login-card h1 {
      font-size: var(--md-sys-typescale-headline-small-font-size, 24px);
      color: var(--md-sys-color-on-surface);
      margin-bottom: 8px;
    }
    .form-fields {
      display: flex;
      flex-direction: column;
      gap: 16px;
      margin: 24px 0;
    }
    md-filled-text-field, md-outlined-text-field {
      width: 100%;
    }
    .actions {
      display: flex;
      justify-content: flex-end;
      gap: 8px;
    }
  </style>
</head>
<body>
  <div class="login-card">
    <h1>Sign in</h1>
    <p style="color: var(--md-sys-color-on-surface-variant, #44474E);">
      Use your account to continue
    </p>
    <div class="form-fields">
      <md-outlined-text-field
        label="Email"
        type="email"
        autocomplete="email"
        required
      >
        <md-icon slot="leading-icon">email</md-icon>
      </md-outlined-text-field>
      <md-outlined-text-field
        label="Password"
        type="password"
        autocomplete="current-password"
        required
      >
        <md-icon slot="leading-icon">lock</md-icon>
      </md-outlined-text-field>
    </div>
    <div class="actions">
      <md-text-button>Forgot password?</md-text-button>
      <md-filled-button type="submit">Sign in</md-filled-button>
    </div>
  </div>

  <script type="module">
    import '@material/web/button/filled-button.js';
    import '@material/web/button/text-button.js';
    import '@material/web/textfield/outlined-text-field.js';
    import '@material/web/icon/icon.js';
  </script>
</body>
</html>
```

### Theme Generation from Seed Color

```js
// theme.js — generate MD3 tonal palette from seed color
import { argbFromHex, themeFromSourceColor, applyTheme } from '@material/material-color-utilities';

const theme = themeFromSourceColor(argbFromHex('#1A73E8'), [
  { name: 'brand', value: argbFromHex('#1A73E8'), blend: true },
]);

// Apply to document (light or dark)
applyTheme(theme, { target: document.body, dark: false });
```

```css
/* Or manually define a baseline light theme */
:root {
  --md-sys-color-primary: #1A73E8;
  --md-sys-color-on-primary: #FFFFFF;
  --md-sys-color-primary-container: #D3E3FD;
  --md-sys-color-on-primary-container: #001D35;
  --md-sys-color-secondary: #565F71;
  --md-sys-color-on-secondary: #FFFFFF;
  --md-sys-color-secondary-container: #DAE2F9;
  --md-sys-color-on-secondary-container: #131C2C;
  --md-sys-color-tertiary: #715573;
  --md-sys-color-on-tertiary: #FFFFFF;
  --md-sys-color-error: #BA1A1A;
  --md-sys-color-on-error: #FFFFFF;
  --md-sys-color-error-container: #FFDAD6;
  --md-sys-color-background: #FDFCFF;
  --md-sys-color-on-background: #1A1C1E;
  --md-sys-color-surface: #FDFCFF;
  --md-sys-color-on-surface: #1A1C1E;
  --md-sys-color-surface-variant: #E0E2EC;
  --md-sys-color-on-surface-variant: #44474E;
  --md-sys-color-outline: #74777F;
  --md-sys-color-outline-variant: #C4C6D0;
  --md-sys-color-shadow: #000000;
  --md-sys-color-scrim: #000000;
  --md-sys-color-inverse-surface: #2F3033;
  --md-sys-color-inverse-on-surface: #F1F0F4;
  --md-sys-color-inverse-primary: #A8C8FA;
}

/* Dark theme override */
[data-theme="dark"] {
  --md-sys-color-primary: #A8C8FA;
  --md-sys-color-on-primary: #003062;
  --md-sys-color-primary-container: #004589;
  --md-sys-color-on-primary-container: #D3E3FD;
  --md-sys-color-surface: #1A1C1E;
  --md-sys-color-on-surface: #E3E2E6;
  --md-sys-color-background: #1A1C1E;
  --md-sys-color-on-background: #E3E2E6;
}
```

### Responsive App Shell with Navigation

```html
<!-- Responsive shell: nav bar (mobile) → nav rail (tablet) → nav drawer (desktop) -->
<style>
  .app-shell {
    display: grid;
    grid-template-areas: "content" "nav";
    grid-template-rows: 1fr auto;
    min-height: 100dvh;
  }

  /* Tablet: 600px+ */
  @media (min-width: 600px) {
    .app-shell {
      grid-template-areas: "nav content";
      grid-template-columns: auto 1fr;
      grid-template-rows: 1fr;
    }
    md-navigation-bar { display: none; }
    md-navigation-rail { display: flex; }
  }

  /* Desktop: 1240px+ */
  @media (min-width: 1240px) {
    .app-shell {
      grid-template-columns: 360px 1fr;
    }
    md-navigation-rail { display: none; }
    md-navigation-drawer { display: block; }
  }

  .content {
    grid-area: content;
    padding: 16px;
  }

  /* Default: hide rail and drawer */
  md-navigation-rail,
  md-navigation-drawer { display: none; }
</style>

<div class="app-shell">
  <!-- Mobile bottom nav bar -->
  <md-navigation-bar>
    <md-navigation-tab label="Home" active>
      <md-icon slot="active-icon">home</md-icon>
      <md-icon slot="inactive-icon">home</md-icon>
    </md-navigation-tab>
    <md-navigation-tab label="Search">
      <md-icon slot="active-icon">search</md-icon>
      <md-icon slot="inactive-icon">search</md-icon>
    </md-navigation-tab>
    <md-navigation-tab label="Library">
      <md-icon slot="active-icon">library_books</md-icon>
      <md-icon slot="inactive-icon">library_books</md-icon>
    </md-navigation-tab>
  </md-navigation-bar>

  <!-- Tablet navigation rail (shown via media query) -->
  <md-navigation-rail>
    <md-navigation-tab label="Home" active>
      <md-icon slot="active-icon">home</md-icon>
      <md-icon slot="inactive-icon">home</md-icon>
    </md-navigation-tab>
    <md-navigation-tab label="Search">
      <md-icon slot="active-icon">search</md-icon>
      <md-icon slot="inactive-icon">search</md-icon>
    </md-navigation-tab>
  </md-navigation-rail>

  <!-- Desktop navigation drawer (shown via media query) -->
  <md-navigation-drawer opened>
    <div slot="headline">My App</div>
    <md-list>
      <md-list-item type="link" href="/home">
        <md-icon slot="start">home</md-icon>
        Home
      </md-list-item>
      <md-list-item type="link" href="/search">
        <md-icon slot="start">search</md-icon>
        Search
      </md-list-item>
    </md-list>
  </md-navigation-drawer>

  <main class="content">
    <!-- Page content -->
  </main>
</div>
```

---

## Flutter Implementation

```dart
// pubspec.yaml
// dependencies:
//   flutter:
//     sdk: flutter
//   material_color_utilities: ^0.8.0

import 'package:flutter/material.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    // Generate MD3 theme from seed color
    final colorScheme = ColorScheme.fromSeed(
      seedColor: const Color(0xFF1A73E8),
      brightness: Brightness.light,
    );

    return MaterialApp(
      theme: ThemeData(
        useMaterial3: true,                  // REQUIRED: enable MD3
        colorScheme: colorScheme,
        typography: Typography.material2021(),
      ),
      darkTheme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1A73E8),
          brightness: Brightness.dark,
        ),
      ),
      home: const LoginScreen(),
    );
  }
}

class LoginScreen extends StatelessWidget {
  const LoginScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 400),
          child: Card(
            // MD3 Card uses surface container color automatically
            child: Padding(
              padding: const EdgeInsets.all(32),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Text(
                    'Sign in',
                    style: theme.textTheme.headlineSmall,
                  ),
                  const SizedBox(height: 24),
                  const TextField(
                    decoration: InputDecoration(
                      labelText: 'Email',
                      prefixIcon: Icon(Icons.email_outlined),
                      border: OutlineInputBorder(),
                    ),
                    keyboardType: TextInputType.emailAddress,
                  ),
                  const SizedBox(height: 16),
                  const TextField(
                    decoration: InputDecoration(
                      labelText: 'Password',
                      prefixIcon: Icon(Icons.lock_outlined),
                      border: OutlineInputBorder(),
                    ),
                    obscureText: true,
                  ),
                  const SizedBox(height: 24),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.end,
                    children: [
                      TextButton(
                        onPressed: () {},
                        child: const Text('Forgot password?'),
                      ),
                      const SizedBox(width: 8),
                      FilledButton(
                        onPressed: () {},
                        child: const Text('Sign in'),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
```

### Flutter Responsive Navigation (Adaptive Shell)

```dart
import 'package:flutter/material.dart';

class AdaptiveShell extends StatefulWidget {
  const AdaptiveShell({super.key});

  @override
  State<AdaptiveShell> createState() => _AdaptiveShellState();
}

class _AdaptiveShellState extends State<AdaptiveShell> {
  int _selectedIndex = 0;

  static const _destinations = [
    NavigationDestination(icon: Icon(Icons.home_outlined), selectedIcon: Icon(Icons.home), label: 'Home'),
    NavigationDestination(icon: Icon(Icons.search_outlined), selectedIcon: Icon(Icons.search), label: 'Search'),
    NavigationDestination(icon: Icon(Icons.library_books_outlined), selectedIcon: Icon(Icons.library_books), label: 'Library'),
  ];

  @override
  Widget build(BuildContext context) {
    final width = MediaQuery.sizeOf(context).width;

    // Compact < 600: NavigationBar (bottom)
    if (width < 600) {
      return Scaffold(
        body: _buildBody(),
        bottomNavigationBar: NavigationBar(
          selectedIndex: _selectedIndex,
          onDestinationSelected: (i) => setState(() => _selectedIndex = i),
          destinations: _destinations,
        ),
      );
    }

    // Medium 600–1239: NavigationRail (side)
    if (width < 1240) {
      return Scaffold(
        body: Row(
          children: [
            NavigationRail(
              selectedIndex: _selectedIndex,
              onDestinationSelected: (i) => setState(() => _selectedIndex = i),
              labelType: NavigationRailLabelType.all,
              destinations: _destinations.map((d) => NavigationRailDestination(
                icon: d.icon,
                selectedIcon: d.selectedIcon ?? d.icon,
                label: Text(d.label),
              )).toList(),
            ),
            const VerticalDivider(thickness: 1, width: 1),
            Expanded(child: _buildBody()),
          ],
        ),
      );
    }

    // Expanded 1240+: NavigationDrawer (side)
    return Scaffold(
      body: Row(
        children: [
          NavigationDrawer(
            selectedIndex: _selectedIndex,
            onDestinationSelected: (i) => setState(() => _selectedIndex = i),
            children: [
              const Padding(
                padding: EdgeInsets.fromLTRB(28, 16, 16, 10),
                child: Text('My App', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
              ),
              ..._destinations.map((d) => NavigationDrawerDestination(
                icon: d.icon,
                selectedIcon: d.selectedIcon ?? d.icon,
                label: Text(d.label),
              )),
            ],
          ),
          Expanded(child: _buildBody()),
        ],
      ),
    );
  }

  Widget _buildBody() {
    return const Center(child: Text('Content area'));
  }
}
```

---

## Jetpack Compose Implementation

```kotlin
// build.gradle.kts
// implementation("androidx.compose.material3:material3:1.2.1")
// implementation("androidx.compose.material3:material3-adaptive-navigation-suite:1.0.0")

import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.foundation.layout.*
import androidx.compose.ui.Modifier

// 1. Set up MD3 theme
@Composable
fun AppTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = true,        // Android 12+ wallpaper colors
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context)
            else dynamicLightColorScheme(context)
        }
        darkTheme -> darkColorScheme(primary = Color(0xFFA8C8FA))
        else -> lightColorScheme(primary = Color(0xFF1A73E8))
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography(),       // MD3 type scale
        content = content
    )
}

// 2. Login form
@Composable
fun LoginScreen() {
    var email by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = MaterialTheme.colorScheme.background
    ) {
        Box(contentAlignment = Alignment.Center) {
            Card(
                modifier = Modifier.widthIn(max = 400.dp).padding(16.dp)
            ) {
                Column(modifier = Modifier.padding(32.dp)) {
                    Text("Sign in", style = MaterialTheme.typography.headlineSmall)
                    Spacer(Modifier.height(24.dp))
                    OutlinedTextField(
                        value = email,
                        onValueChange = { email = it },
                        label = { Text("Email") },
                        leadingIcon = { Icon(Icons.Outlined.Email, contentDescription = null) },
                        modifier = Modifier.fillMaxWidth()
                    )
                    Spacer(Modifier.height(16.dp))
                    OutlinedTextField(
                        value = password,
                        onValueChange = { password = it },
                        label = { Text("Password") },
                        leadingIcon = { Icon(Icons.Outlined.Lock, contentDescription = null) },
                        visualTransformation = PasswordVisualTransformation(),
                        modifier = Modifier.fillMaxWidth()
                    )
                    Spacer(Modifier.height(24.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.End
                    ) {
                        TextButton(onClick = {}) { Text("Forgot password?") }
                        Spacer(Modifier.width(8.dp))
                        Button(onClick = {}) { Text("Sign in") }
                    }
                }
            }
        }
    }
}
```

---

## MD3 Compliance Audit

Run the audit to get a scored report across 10 categories:

```bash
/material-3 audit ./src
/material-3 audit https://yourapp.com
```

### Audit Categories

| # | Category | What's checked |
|---|---|---|
| 1 | **Color** | Using `--md-sys-color-*` tokens, not raw hex; color roles correctly applied |
| 2 | **Typography** | Using `--md-sys-typescale-*` tokens; correct scale roles (Display/Headline/Title/Body/Label) |
| 3 | **Shape** | Using `--md-sys-shape-corner-*` tokens; shape tier matches component spec |
| 4 | **Elevation** | Using `--md-sys-elevation-level*`; correct level per component; tonal surface overlays in dark mode |
| 5 | **Components** | Using `@material/web` elements (not custom copies); correct attributes set |
| 6 | **Layout** | Responsive breakpoints (compact/medium/expanded); correct margin/column counts |
| 7 | **Navigation** | Correct nav component for screen size; nav bar ↔ rail ↔ drawer transitions |
| 8 | **Motion** | MD3 motion tokens; M3 Expressive spring motion where applicable |
| 9 | **Accessibility** | ARIA labels; color contrast ≥ 4.5:1; focus indicators; touch targets ≥ 48dp |
| 10 | **Theming** | Dynamic color support; dark mode; theme tokens centralized |

---

## Component Quick Reference

### Buttons

```html
<md-filled-button>Filled (primary action)</md-filled-button>
<md-filled-tonal-button>Filled Tonal (secondary)</md-filled-tonal-button>
<md-outlined-button>Outlined (medium emphasis)</md-outlined-button>
<md-text-button>Text (low emphasis)</md-text-button>
<md-elevated-button>Elevated</md-elevated-button>
```

### Text Fields

```html
<md-filled-text-field label="Name" value="" type="text"></md-filled-text-field>
<md-outlined-text-field label="Email" type="email" required></md-outlined-text-field>
<!-- Supporting text, error, leading/trailing icons -->
<md-outlined-text-field label="Search" type="search">
  <md-icon slot="leading-icon">search</md-icon>
</md-outlined-text-field>
```

### FAB

```html
<md-fab aria-label="Add item">
  <md-icon slot="icon">add</md-icon>
</md-fab>
<md-fab size="small" aria-label="Add"><md-icon slot="icon">add</md-icon></md-fab>
<md-fab size="large" aria-label="Add"><md-icon slot="icon">add</md-icon></md-fab>
<md-branded-fab aria-label="Edit">
  <md-icon slot="icon">edit</md-icon>
</md-branded-fab>
```

### Dialog

```html
<md-dialog id="confirm-dialog">
  <div slot="headline">Delete item?</div>
  <div slot="content">This action cannot be undone.</div>
  <div slot="actions">
    <md-text-button form="confirm-form" value="cancel">Cancel</md-text-button>
    <md-filled-button form="confirm-form" value="delete">Delete</md-filled-button>
  </div>
  <form id="confirm-form" slot="content" method="dialog"></form>
</md-dialog>

<script type="module">
  document.querySelector('#confirm-dialog').show();
</script>
```

### Chips

```html
<md-chip-set>
  <md-filter-chip label="Favorites" selected></md-filter-chip>
  <md-filter-chip label="Recent"></md-filter-chip>
  <md-assist-chip label="Turn on lights">
    <md-icon slot="icon">lightbulb</md-icon>
  </md-assist-chip>
  <md-input-chip label="Flutter" removable></md-input-chip>
</md-chip
