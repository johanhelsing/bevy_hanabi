//! GPU tests of the particle bookkeeping of a child effect fed by its
//! parent's spawn events, with and without ribbons.
//!
//! Each test runs a headless app at a fixed frame step under
//! [`BookkeepingProbePlugin`], which once a second reads back every effect's
//! alive and dead lists and counts the slots lost or listed twice. Each lost
//! slot is one the effect never spawns into again.

use std::time::Duration;

use bevy::{
    app::PluginsState,
    camera::RenderTarget,
    gilrs::GilrsPlugin,
    log::LogPlugin,
    prelude::*,
    render::{pipelined_rendering::PipelinedRenderingPlugin, render_resource::TextureFormat},
    time::TimeUpdateStrategy,
    window::ExitCondition,
    winit::WinitPlugin,
};

use crate::{
    Attribute, BookkeepingProbePlugin, BookkeepingSample, BookkeepingSamples, EffectAsset,
    EffectParent, EmitSpawnEventModifier, EventEmitCondition, ExprWriter, HanabiPlugin,
    InheritAttributeModifier, MotionIntegration, ParticleEffect, SetAttributeModifier,
    SetPositionCircleModifier, SetVelocityCircleModifier, ShapeDimension, SimulationCondition,
    SpawnerSettings,
};

/// Settings of one run.
#[derive(Debug, Clone, Copy)]
struct Setup {
    /// Parent particles born per second.
    heads_per_second: f32,
    /// Lifetime of a parent particle, in seconds.
    head_lifetime: f32,
    /// Lifetime of a child particle, in seconds. Each parent particle emits
    /// one child particle per frame.
    body_lifetime: f32,
    /// Capacity of the child effect.
    body_capacity: u32,
    /// Whether the child particles carry a `RIBBON_ID`.
    ribbon: bool,
    /// Pairs of parent and child effects spawned from the same two assets.
    /// The last child is read back.
    instances: u32,
    /// Whether each pair gets assets of its own, as two different emitters
    /// with trails do, rather than sharing one pair of assets.
    own_assets: bool,
    /// Capacity of the parent effect, or twice the parents alive if `None`.
    head_capacity: Option<u32>,
    /// Pairs alive before the measured pairs and despawned once those are
    /// running, as a game replaces its emitters: the new trails take their
    /// spawn event rows while the old ones still hold theirs.
    retired: u32,
    /// Frame times in seconds, cycled through.
    frame_secs: &'static [f64],
}

fn head_effect(setup: Setup) -> EffectAsset {
    let writer = ExprWriter::new();
    let position = SetPositionCircleModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        axis: writer.lit(Vec3::Z).expr(),
        radius: writer.lit(1.).expr(),
        dimension: ShapeDimension::Volume,
    };
    let velocity = SetVelocityCircleModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        axis: writer.lit(Vec3::Z).expr(),
        speed: writer.lit(2.).expr(),
    };
    let age = SetAttributeModifier::new(Attribute::AGE, writer.lit(0.).expr());
    let lifetime =
        SetAttributeModifier::new(Attribute::LIFETIME, writer.lit(setup.head_lifetime).expr());
    // A unique value per parent particle, read by the child as its ribbon id.
    let ribbon_id = SetAttributeModifier::new(
        Attribute::U32_0,
        writer.attr(Attribute::PARTICLE_COUNTER).expr(),
    );
    let trail = EmitSpawnEventModifier {
        condition: EventEmitCondition::Always,
        count: writer.lit(1u32).expr(),
        child_index: 0,
    };
    let capacity = setup
        .head_capacity
        .unwrap_or((setup.heads_per_second * setup.head_lifetime * 2.) as u32);
    EffectAsset::new(
        capacity,
        SpawnerSettings::rate(setup.heads_per_second.into()),
        writer.finish(),
    )
    .with_name("heads")
    .with_simulation_condition(SimulationCondition::Always)
    .init(position)
    .init(velocity)
    .init(age)
    .init(lifetime)
    .init(ribbon_id)
    .update(trail)
}

fn body_effect(setup: Setup) -> EffectAsset {
    let writer = ExprWriter::new();
    let position = InheritAttributeModifier::new(Attribute::POSITION);
    let age = SetAttributeModifier::new(Attribute::AGE, writer.lit(0.).expr());
    let lifetime =
        SetAttributeModifier::new(Attribute::LIFETIME, writer.lit(setup.body_lifetime).expr());
    let ribbon_id = setup.ribbon.then(|| {
        SetAttributeModifier::new(
            Attribute::RIBBON_ID,
            writer.parent_attr(Attribute::U32_0).expr(),
        )
    });
    let mut asset = EffectAsset::new(
        setup.body_capacity,
        SpawnerSettings::default(),
        writer.finish(),
    )
    .with_name("bodies")
    .with_simulation_condition(SimulationCondition::Always)
    .with_motion_integration(MotionIntegration::None)
    .init(position)
    .init(age)
    .init(lifetime);
    if let Some(ribbon_id) = ribbon_id {
        asset = asset.init(ribbon_id);
    }
    asset
}

/// Run the parent and child effects for `frames` frames and return the
/// readbacks.
/// The readbacks of a run, and the main-world entities of its child effects.
struct Run {
    samples: Vec<BookkeepingSample>,
    children: Vec<Entity>,
}

/// A headless app with hanabi, the bookkeeping probe and a camera, which
/// hanabi needs before it allocates its buffers.
fn headless_app() -> App {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                close_when_requested: false,
                ..default()
            })
            .set(LogPlugin {
                filter: "wgpu=error,bevy_hanabi=warn".into(),
                ..default()
            })
            .disable::<WinitPlugin>()
            .disable::<GilrsPlugin>()
            .disable::<PipelinedRenderingPlugin>(),
    )
    .add_plugins((HanabiPlugin, BookkeepingProbePlugin::default()));

    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
    }
    app.finish();
    app.cleanup();

    let world = app.world_mut();
    // hanabi allocates its buffers only once a view exists.
    let target = world
        .resource_mut::<Assets<Image>>()
        .add(Image::new_target_texture(
            64,
            64,
            TextureFormat::Rgba8UnormSrgb,
            None,
        ));
    world.spawn((
        Camera3d::default(),
        RenderTarget::Image(target.into()),
        Transform::from_xyz(0., 0., 10.).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    app
}

fn run(setup: Setup, frames: u32) -> Run {
    let mut app = headless_app();
    let world = app.world_mut();
    let mut assets = world.resource_mut::<Assets<EffectAsset>>();
    let (retired_head, retired_body) = (
        assets.add(head_effect(setup)),
        assets.add(body_effect(setup)),
    );
    let retired: Vec<_> = (0..setup.retired)
        .flat_map(|_| {
            let parent = world.spawn(ParticleEffect::new(retired_head.clone())).id();
            let child = world
                .spawn((
                    ParticleEffect::new(retired_body.clone()),
                    EffectParent::new(parent),
                ))
                .id();
            [child, parent]
        })
        .collect();
    for _ in 0..10 {
        app.update();
    }
    let world = app.world_mut();
    let mut assets = world.resource_mut::<Assets<EffectAsset>>();
    let shared = (
        assets.add(head_effect(setup)),
        assets.add(body_effect(setup)),
    );
    let pairs: Vec<_> = (0..setup.instances)
        .map(|_| {
            if setup.own_assets {
                (
                    assets.add(head_effect(setup)),
                    assets.add(body_effect(setup)),
                )
            } else {
                shared.clone()
            }
        })
        .collect();
    let mut children = Vec::new();
    for (head, body) in pairs {
        let parent = world.spawn(ParticleEffect::new(head)).id();
        children.push(
            world
                .spawn((ParticleEffect::new(body), EffectParent::new(parent)))
                .id(),
        );
    }

    for _ in 0..10 {
        app.update();
    }
    for entity in retired {
        app.world_mut().despawn(entity);
    }

    for frame in 0..frames {
        let secs = setup.frame_secs[frame as usize % setup.frame_secs.len()];
        app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
            secs,
        )));
        app.update();
    }
    Run {
        samples: app.world().resource::<BookkeepingSamples>().get(),
        children,
    }
}

/// Fails when any readback found a slot listed twice or nowhere, or a child
/// fed more than one spawn event per parent particle alive in a frame, or
/// holding more points than that feeds it, each living `body_lifetime`.
fn assert_bookkeeping(setup: Setup, run: &Run) {
    let samples = &run.samples;
    assert!(!samples.is_empty(), "no effect was read back");
    let heads = setup.heads_per_second * setup.head_lifetime;
    let children: Vec<_> = samples
        .iter()
        .filter(|s| run.children.contains(&s.entity))
        .collect();
    assert!(!children.is_empty(), "no child effect was read back");
    let overfed: Vec<_> = children
        .iter()
        .filter(|s| {
            s.events
                .is_none_or(|events| events as f32 > heads * 1.5 + 16.)
        })
        .collect();
    assert!(
        overfed.is_empty(),
        "child effects fed more than about {heads} events a frame: {overfed:?}"
    );
    let shortest_frame = setup.frame_secs.iter().copied().fold(f64::MAX, f64::min);
    let expected = heads * setup.body_lifetime / shortest_frame as f32;
    let overfull: Vec<_> = children
        .iter()
        .filter(|s| s.alive_count as f32 > expected * 1.5 + 64.)
        .collect();
    assert!(
        overfull.is_empty(),
        "child effects hold more than about {expected} points: {overfull:?}"
    );
    let bad: Vec<_> = samples
        .iter()
        .filter(|s| s.duplicated > 0 || s.leaked > 0)
        .collect();
    assert!(bad.is_empty(), "slots lost or listed twice: {bad:?}");
}

/// The shape of the weld sparks' trails in Cargo Space: 50 parents alive, one
/// child point each per frame for 0.3 s, in one sort block.
const TRAILS: Setup = Setup {
    heads_per_second: 25.,
    head_lifetime: 2.,
    body_lifetime: 0.3,
    body_capacity: 1024,
    ribbon: true,
    instances: 1,
    own_assets: false,
    head_capacity: None,
    retired: 0,
    frame_secs: &[1. / 60.],
};

/// Run `setup` for ten seconds and check it.
fn check(setup: Setup) {
    assert_bookkeeping(setup, &run(setup, 60 * 10));
}

/// Five minutes of a trail spawned while older emitters ran: the point count
/// must hold steady to the end.
#[test]
#[ignore = "slow: five minutes of frames, about fifteen minutes to run"]
fn a_trail_after_retired_emitters_holds_steady_for_five_minutes() {
    let setup = Setup {
        retired: 2,
        body_capacity: 32768,
        ..TRAILS
    };
    let run = run(setup, 60 * 60 * 5);
    assert_bookkeeping(setup, &run);
    let counts: Vec<_> = run
        .samples
        .iter()
        .filter(|s| run.children.contains(&s.entity))
        .map(|s| s.alive_count)
        .skip(5)
        .collect();
    let (low, high) = (counts.iter().min().unwrap(), counts.iter().max().unwrap());
    println!("child points over five minutes: {low} to {high}");
    assert!(high - low <= 64, "point count drifts from {low} to {high}");
}

#[test]
fn ribbon_trails_keep_every_slot() {
    check(TRAILS);
}

#[test]
fn dot_trails_keep_every_slot() {
    check(Setup {
        ribbon: false,
        ..TRAILS
    });
}

#[test]
fn full_trail_keeps_every_slot() {
    check(Setup {
        body_capacity: 512,
        ..TRAILS
    });
}

#[test]
fn trails_keep_every_slot_over_varied_frame_times() {
    check(Setup {
        frame_secs: &[1. / 144., 1. / 30., 1. / 60., 0.1, 1. / 240.],
        ..TRAILS
    });
}

#[test]
fn two_trails_of_one_asset_keep_every_slot() {
    check(Setup {
        instances: 2,
        ..TRAILS
    });
}

#[test]
fn a_trail_over_many_sort_blocks_keeps_every_slot() {
    check(Setup {
        heads_per_second: 200.,
        body_capacity: 8192,
        ..TRAILS
    });
}

/// The weld's shape: two emitters, each its own parent and trail assets,
/// parents in a roomy effect and trails with room to spare.
#[test]
fn two_trails_of_their_own_assets_hold_steady() {
    check(Setup {
        instances: 2,
        own_assets: true,
        head_capacity: Some(2048),
        body_capacity: 32768,
        ..TRAILS
    });
}

/// Emitters come and go in a game: a trail spawned while older ones still
/// run must stay fed one point per parent per frame once they are gone.
#[test]
fn a_trail_after_retired_emitters_holds_steady() {
    check(Setup {
        retired: 2,
        ..TRAILS
    });
}

/// A parent at full capacity gets a trail point for every particle each
/// frame: its child's spawn event slice is sized from the parent's capacity,
/// so none of the events is dropped. The parent holds more particles than the
/// fixed slice of 16384 events the child had before.
#[test]
fn a_full_parent_gets_a_trail_point_for_every_particle() {
    const PARENTS: u32 = 20_000;
    let frame_secs: f32 = 1. / 60.;
    let mut app = headless_app();
    let world = app.world_mut();
    let writer = ExprWriter::new();
    let position = SetAttributeModifier::new(Attribute::POSITION, writer.lit(Vec3::ZERO).expr());
    let age = SetAttributeModifier::new(Attribute::AGE, writer.lit(0.).expr());
    let lifetime = SetAttributeModifier::new(Attribute::LIFETIME, writer.lit(1000.).expr());
    let trail = EmitSpawnEventModifier {
        condition: EventEmitCondition::Always,
        count: writer.lit(1u32).expr(),
        child_index: 0,
    };
    let head = EffectAsset::new(
        PARENTS,
        SpawnerSettings::once((PARENTS as f32).into()),
        writer.finish(),
    )
    .with_name("full heads")
    .with_simulation_condition(SimulationCondition::Always)
    .with_motion_integration(MotionIntegration::None)
    .init(position)
    .init(age)
    .init(lifetime)
    .update(trail);
    // Each point lives a frame and a half, so the points alive after an
    // update pass are the ones born from a single frame's events.
    let body = body_effect(Setup {
        body_lifetime: 1.5 * frame_secs,
        body_capacity: 2 * PARENTS,
        ribbon: false,
        ..TRAILS
    });
    let mut assets = world.resource_mut::<Assets<EffectAsset>>();
    let (head, body) = (assets.add(head), assets.add(body));
    let parent = world.spawn(ParticleEffect::new(head)).id();
    let child = world
        .spawn((ParticleEffect::new(body), EffectParent::new(parent)))
        .id();

    app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
        frame_secs,
    )));
    for _ in 0..60 * 3 {
        app.update();
    }

    let samples = app.world().resource::<BookkeepingSamples>().get();
    let parents: Vec<_> = samples
        .iter()
        .filter(|s| s.entity == parent)
        .map(|s| s.alive_count)
        .collect();
    let points: Vec<_> = samples
        .iter()
        .filter(|s| s.entity == child)
        .map(|s| s.alive_count)
        .collect();
    println!("parents alive: {parents:?}, points alive: {points:?}");
    assert!(
        parents.len() > 1 && parents.iter().skip(1).all(|&n| n == PARENTS),
        "the parent is not full: {parents:?}"
    );
    assert!(
        points.len() > 1 && points.iter().skip(1).all(|&n| n == PARENTS),
        "the {PARENTS} parents do not get a point each: {points:?}"
    );
}
