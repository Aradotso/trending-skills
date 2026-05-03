```markdown
---
name: compose-performance-skills
description: Install and use the skydoves/compose-performance-skills agent skill library to diagnose and fix Jetpack Compose performance issues including stability, recomposition, lazy layouts, modifiers, side effects, and build configuration.
triggers:
  - "my composable recomposes too often"
  - "LazyColumn drops frames during scroll"
  - "diagnose Compose stability issues"
  - "fix unnecessary recomposition in Jetpack Compose"
  - "optimize Compose performance"
  - "Compose unstable parameters compiler report"
  - "baseline profile for Compose app"
  - "install compose performance skills"
---

# Compose Performance Skills

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A curated library of Agent Skills (SKILL.md files) for Jetpack Compose performance. Each skill teaches an AI coding agent to diagnose and fix one specific Compose performance problem—stability, recomposition, lazy layouts, custom modifiers, side effects, baseline profiles, R8, and hot reload.

## What This Library Does

- **26 skills** organized into 9 categories: `stability`, `recomposition`, `lists`, `modifiers`, `side-effects`, `measurement`, `build`, `audit`, `hot-reload`
- Each `SKILL.md` is operational instructions for an LLM—terse, imperative, with RIGHT/WRONG code pairs and a verification checklist
- Skills chain together: a diagnostic skill names the fix skills it hands off to; a fix skill states which diagnostic skill produced its input
- Grounded in primary sources: Android Developers docs, Compose compiler internals, AndroidX release notes

## Installation

### Claude Code (Recommended)

```bash
# Clone to stable location
git clone https://github.com/skydoves/compose-performance-skills.git \
  ~/.claude/skills-sources/compose-performance-skills

# Run idempotent install script (symlinks each skill into ~/.claude/skills/<slug>/)
~/.claude/skills-sources/compose-performance-skills/scripts/install-skills.sh
```

Custom target directory:
```bash
./scripts/install-skills.sh /path/to/agent/skills
```

Uninstall:
```bash
./scripts/install-skills.sh --uninstall
```

Restart Claude Code after install. Mention a Compose performance symptom and Claude Code matches trigger vocabulary in skill frontmatter to load the relevant `SKILL.md` automatically.

### Android Studio Agent Mode / Gemini (Project-Local)

```bash
cd <your-android-project>

git clone https://github.com/skydoves/compose-performance-skills.git \
  .compose-performance-skills-source

./.compose-performance-skills-source/scripts/install-skills.sh .agent/skills
```

Add `.compose-performance-skills-source/` to `.gitignore`. Update skills with `git pull` inside the source directory.

### Claude.ai / Anthropic API

Upload individual `SKILL.md` files as Agent Skill attachments in a Claude.ai workspace, or inject them into the system prompt for direct API use. Skills are self-contained Markdown—no external runtime required.

## Directory Layout

```
compose-performance-skills/
├── README.md
├── INDEX.md                        # symptom → skill lookup table
├── scripts/
│   └── install-skills.sh
└── <category>/<slug>/
    ├── SKILL.md                    # the skill (required)
    └── references/
        └── <topic>.md              # supplemental material (one level deep)
```

**Categories and slugs:**

| Category | Key Skills |
|---|---|
| `stability/` | `diagnosing-compose-stability`, `stabilizing-compose-types`, `configuring-stability-config` |
| `recomposition/` | `deferring-state-reads`, `applying-derived-state-of`, `tracing-recomposition` |
| `lists/` | `optimizing-lazy-layouts`, `stable-keys-content-type`, `lazy-layout-prefetch` |
| `modifiers/` | `migrating-to-modifier-node`, `graphics-layer-animated-reads` |
| `side-effects/` | `collecting-flows-safely`, `memoizing-lambdas-strong-skipping` |
| `measurement/` | `running-macrobenchmark`, `generating-baseline-profiles` |
| `build/` | `enabling-r8-full-mode`, `enabling-strong-skipping`, `ci-stability-validation` |
| `audit/` | `end-to-end-performance-audit` |
| `hot-reload/` | `compose-hot-reload-hotswan` |

## Key Skill Workflows

### Diagnosing Stability (`stability/diagnosing-compose-stability/SKILL.md`)

Triggered by: "recomposition spike", "unstable parameters", "composable won't skip"

**What it does:**
1. Enables Compose Compiler reports in `release` build type
2. Parses `<module>-composables.txt` for `unstable` parameters
3. Chains to `stabilizing-compose-types` for the fix

```kotlin
// build.gradle.kts – enable compiler reports
android {
    buildTypes {
        release {
            // Compose compiler metrics output
        }
    }
}

tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile>().configureEach {
    compilerOptions {
        freeCompilerArgs.addAll(
            "-P",
            "plugin:androidx.compose.compiler.plugins.kotlin:reportsDestination=" +
                layout.buildDirectory.dir("compose_metrics").get().asFile.absolutePath,
            "-P",
            "plugin:androidx.compose.compiler.plugins.kotlin:metricsDestination=" +
                layout.buildDirectory.dir("compose_metrics").get().asFile.absolutePath
        )
    }
}
```

Run the report:
```bash
./gradlew :app:assembleRelease
cat app/build/compose_metrics/app_release-composables.txt | grep "unstable"
```

### Stabilizing Compose Types (`stability/stabilizing-compose-types/SKILL.md`)

```kotlin
// WRONG – List<T> is unstable
@Composable
fun ItemList(items: List<Item>) { ... }

// RIGHT – use ImmutableList from kotlinx.collections.immutable
import kotlinx.collections.immutable.ImmutableList
import kotlinx.collections.immutable.persistentListOf

@Composable
fun ItemList(items: ImmutableList<Item>) { ... }

// In ViewModel
val items: StateFlow<ImmutableList<Item>> = _items.asStateFlow()
```

```kotlin
// WRONG – data class with unstable field
data class UiState(
    val items: List<Item>,          // unstable
    val callback: () -> Unit        // unstable
)

// RIGHT – @Stable annotation + immutable collections
@Stable
data class UiState(
    val items: ImmutableList<Item>,
    val onItemClick: (Item) -> Unit
)
```

### Deferring State Reads (`recomposition/deferring-state-reads/SKILL.md`)

```kotlin
// WRONG – offset read triggers recomposition of the entire composable
@Composable
fun AnimatedBox(scrollState: ScrollState) {
    val offset = scrollState.value  // read here = recompose entire function
    Box(Modifier.offset(y = offset.dp)) { ... }
}

// RIGHT – defer read into modifier lambda (layout phase only)
@Composable
fun AnimatedBox(scrollState: ScrollState) {
    Box(
        Modifier.offset {                    // read deferred to layout phase
            IntOffset(0, scrollState.value)
        }
    ) { ... }
}

// RIGHT – defer into graphicsLayer lambda (draw phase only)
@Composable
fun FadingBox(alpha: State<Float>) {
    Box(
        Modifier.graphicsLayer {             // read deferred to draw phase
            this.alpha = alpha.value
        }
    ) { ... }
}
```

### Optimizing Lazy Layouts (`lists/optimizing-lazy-layouts/SKILL.md`)

```kotlin
// WRONG – no keys, no contentType
LazyColumn {
    items(itemList) { item ->
        ItemCard(item)
    }
}

// RIGHT – stable keys + contentType
LazyColumn {
    items(
        items = itemList,
        key = { item -> item.id },              // stable, unique key
        contentType = { item -> item::class }   // group similar items
    ) { item ->
        ItemCard(item)
    }
}
```

```kotlin
// WRONG – lambda captures unstable reference
LazyColumn {
    items(items, key = { it.id }) { item ->
        ItemCard(
            item = item,
            onClick = { viewModel.onItemClick(item) }  // new lambda each recomposition
        )
    }
}

// RIGHT – hoist click handler, use rememberUpdatedState if needed
LazyColumn {
    items(items, key = { it.id }) { item ->
        ItemCard(
            item = item,
            onClick = remember(item.id) { { viewModel.onItemClick(item) } }
        )
    }
}
```

### Collecting Flows Safely (`side-effects/collecting-flows-safely/SKILL.md`)

```kotlin
// WRONG – collect in composable body, unstable Flow parameter
@Composable
fun Screen(viewModel: MyViewModel) {
    val state = viewModel.uiFlow.collectAsState()  // collectAsState ignores lifecycle
    // ...
}

// RIGHT – collectAsStateWithLifecycle (lifecycle-aware, Compose 1.2+)
import androidx.lifecycle.compose.collectAsStateWithLifecycle

@Composable
fun Screen(viewModel: MyViewModel) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    // ...
}
```

### Migrating to Modifier.Node (`modifiers/migrating-to-modifier-node/SKILL.md`)

```kotlin
// WRONG – composed { } allocates on every recomposition
fun Modifier.myCustomModifier(color: Color): Modifier = composed {
    val density = LocalDensity.current
    drawBehind {
        drawRect(color)
    }
}

// RIGHT – Modifier.Node skips allocation after first composition
private class MyCustomModifierNode(var color: Color) : DrawModifierNode, Modifier.Node() {
    override fun ContentDrawScope.draw() {
        drawRect(color)
        drawContent()
    }
}

private data class MyCustomModifierElement(val color: Color) :
    ModifierNodeElement<MyCustomModifierNode>() {
    override fun create() = MyCustomModifierNode(color)
    override fun update(node: MyCustomModifierNode) {
        node.color = color
    }
}

fun Modifier.myCustomModifier(color: Color): Modifier = this then MyCustomModifierElement(color)
```

### Enabling Strong Skipping (`build/enabling-strong-skipping/SKILL.md`)

```kotlin
// build.gradle.kts
composeCompiler {
    enableStrongSkippingMode = true   // Compose 1.5+ / Kotlin 1.9+
}
```

Strong skipping allows composables with unstable but equal parameters to skip recomposition using instance equality. **MUST** validate with compiler reports after enabling—some lambdas are memoized automatically but not all.

### Running Macrobenchmark (`measurement/running-macrobenchmark/SKILL.md`)

```kotlin
// benchmark/src/androidTest/kotlin/ScrollBenchmark.kt
@RunWith(AndroidJUnit4::class)
class ScrollBenchmark {
    @get:Rule
    val benchmarkRule = MacrobenchmarkRule()

    @Test
    fun scrollFeed() = benchmarkRule.measureRepeated(
        packageName = "com.example.app",
        metrics = listOf(FrameTimingMetric()),
        compilationMode = CompilationMode.Full(),   // MUST use Full or SpeedProfile
        startupMode = StartupMode.WARM,
        iterations = 5,
        setupBlock = {
            pressHome()
            startActivityAndWait()
        }
    ) {
        val feedList = device.findObject(By.res("feed_list"))
        feedList.fling(Direction.DOWN)
        device.waitForIdle()
    }
}
```

```bash
./gradlew :benchmark:connectedReleaseAndroidTest \
  -P android.testInstrumentationRunnerArguments.androidx.benchmark.output.enable=true
```

**MUST** run benchmarks on a real device in `release` build type with R8 enabled. Debug builds produce misleading results.

## Non-Negotiable Rules (Enforced Across All Skills)

These appear as MUST/MUST NOT directives in every skill body:

1. **Skippability is a diagnostic, not a KPI.** Do not chase 100% skippable composables.
2. **`stability_config.conf` is a contract with the compiler.** Adding a class to the config without making it truly stable causes silent missed recompositions.
3. **`Row`, `Column`, `Box` are inline—not restartable or skippable.** Wrapping them changes recomposition scope.
4. **`Flow` parameters are unstable.** Collect in a `ViewModel` or with `collectAsStateWithLifecycle`. Never pass `Flow` as a composable parameter.
5. **Always measure in release + R8 + real device.** Debug JIT and interpreter noise makes performance data unreliable.

## Symptom → Skill Quick Lookup

| Symptom | Skill |
|---|---|
| Recomposition count spikes on scroll | `stability/diagnosing-compose-stability` |
| `unstable` in compiler report | `stability/stabilizing-compose-types` |
| List with stable data still recomposes | `lists/optimizing-lazy-layouts` |
| Animation jank, graphicsLayer abuse | `modifiers/graphics-layer-animated-reads` |
| `composed {}` in hot path | `modifiers/migrating-to-modifier-node` |
| Flow collected in composable | `side-effects/collecting-flows-safely` |
| Cold startup slow | `measurement/generating-baseline-profiles` |
| Want full audit from scratch | `audit/end-to-end-performance-audit` |

Full table: [`INDEX.md`](https://github.com/skydoves/compose-performance-skills/blob/main/INDEX.md)

## Updating Skills

```bash
# Pull latest skills
cd ~/.claude/skills-sources/compose-performance-skills
git pull

# Re-run install to pick up new skills (idempotent)
./scripts/install-skills.sh
```

## Troubleshooting

**Skills not triggering in Claude Code**
- Verify symlinks exist: `ls ~/.claude/skills/ | grep compose`
- Each skill must be at `~/.claude/skills/<slug>/SKILL.md` (flat, not nested by category)
- Restart Claude Code after install
- Use exact trigger phrases from `INDEX.md` if auto-matching fails

**`install-skills.sh` permission denied**
```bash
chmod +x ~/.claude/skills-sources/compose-performance-skills/scripts/install-skills.sh
```

**Compiler reports empty or missing**
- Confirm you ran `assembleRelease`, not `assembleDebug`
- Check `compilerOptions` block is inside `KotlinCompile` task, not `kotlinOptions`
- Compose compiler plugin version must match Kotlin version (check [compatibility map](https://developer.android.com/jetpack/androidx/releases/compose-kotlin))

**Strong skipping causes incorrect UI**
- Disable with `enableStrongSkippingMode = false` and re-run compiler reports
- Check for composables relying on referential equality of mutable objects—strong skipping uses `==` not `===`

**Benchmark results vary widely**
- Lock CPU clock speed on rooted device or use a Pixel with locked clocks
- Run minimum 5 iterations (`iterations = 5`)
- Use `CompilationMode.Full()` not `CompilationMode.None()` for production-representative results
```
