//! A debug check of the particle bookkeeping of every effect on the GPU.
//!
//! Each effect keeps its particle slots in two lists in the indirect index
//! buffer: the alive list (rows `[0, alive_count)` of the column the update
//! pass wrote this frame) and the dead list (rows `[alive_count, capacity)` of
//! the dead column). Every slot must be in exactly one of them. A slot listed
//! twice was duplicated, and a slot listed nowhere has leaked: it is never
//! updated nor recycled again, and the effect has one slot less to spawn into
//! for the rest of its life.
//!
//! [`BookkeepingProbePlugin`] reads both lists back every few frames, after
//! the frame's commands are submitted, and logs a warning naming each effect
//! that broke the rule. It waits for the GPU on every readback, so it is for
//! debugging only.

use std::sync::{Arc, Mutex};

use bevy::{
    prelude::*,
    render::{
        renderer::{RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue},
        sync_world::MainEntity,
        RenderApp,
    },
};
use wgpu::{BufferDescriptor, BufferUsages, CommandEncoderDescriptor, MapMode, PollType};

use crate::Attribute;

use super::{
    effect_cache::{CachedEffect, EffectCache},
    event::{CachedChildInfo, EventCache, GpuChildInfo},
    CachedEffectMetadata, EffectsMeta, GpuEffectMetadata,
};

/// One readback of one effect's lists.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BookkeepingSample {
    /// Render frame the sample was read on, counted from the plugin's start.
    pub frame: u32,
    /// Main-world entity of the effect.
    pub entity: Entity,
    /// Capacity of the effect, in particles.
    pub capacity: u32,
    /// Particles alive after this frame's update pass.
    pub alive_count: u32,
    /// Slots listed more than once across the alive and dead lists.
    pub duplicated: u32,
    /// Slots listed in neither list.
    pub leaked: u32,
    /// Age of the oldest alive particle, in seconds, for an effect whose
    /// particles have an age.
    pub oldest_age: Option<f32>,
    /// For a child effect, the spawn events its parents emitted this frame,
    /// which it consumes next frame.
    pub events: Option<i32>,
}

/// The readbacks so far, shared by the main and render worlds.
#[derive(Resource, Clone, Default)]
pub struct BookkeepingSamples(Arc<Mutex<Vec<BookkeepingSample>>>);

impl BookkeepingSamples {
    /// A copy of every sample read so far, oldest first.
    pub fn get(&self) -> Vec<BookkeepingSample> {
        self.0.lock().unwrap().clone()
    }
}

/// Reads back every effect's alive and dead lists each `interval` frames and
/// warns when a slot is lost or listed twice (the module doc). Add it after
/// [`HanabiPlugin`]; read the samples from [`BookkeepingSamples`].
///
/// The readback stalls the frame until the GPU is done, so this is a debug
/// tool, never for a shipped build.
///
/// [`HanabiPlugin`]: crate::HanabiPlugin
#[derive(Debug, Clone, Copy)]
pub struct BookkeepingProbePlugin {
    /// Frames between two readbacks.
    pub interval: u32,
}

impl Default for BookkeepingProbePlugin {
    fn default() -> Self {
        Self { interval: 60 }
    }
}

impl Plugin for BookkeepingProbePlugin {
    fn build(&self, app: &mut App) {
        let samples = BookkeepingSamples::default();
        app.insert_resource(samples.clone());
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(samples)
            .insert_resource(ProbeClock {
                interval: self.interval.max(1),
                frame: 0,
            })
            .add_systems(
                RenderGraph,
                probe_bookkeeping.in_set(RenderGraphSystems::Finish),
            );
    }
}

#[derive(Resource)]
struct ProbeClock {
    interval: u32,
    frame: u32,
}

fn probe_bookkeeping(
    samples: Res<BookkeepingSamples>,
    mut clock: ResMut<ProbeClock>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    effects_meta: Res<EffectsMeta>,
    effect_cache: Res<EffectCache>,
    event_cache: Res<EventCache>,
    q_effects: Query<(
        &MainEntity,
        &CachedEffect,
        &CachedEffectMetadata,
        Option<&CachedChildInfo>,
    )>,
) {
    clock.frame += 1;
    if !clock.frame.is_multiple_of(clock.interval) {
        return;
    }
    let Some(metadata_buffer) = effects_meta.effect_metadata_buffer.buffer() else {
        return;
    };
    let metadata_size = size_of::<GpuEffectMetadata>() as u64;
    // Row of the indirect index buffer: ping, pong, dead.
    let row_size = size_of::<[u32; 3]>() as u64;

    // Copy every effect's metadata and rows into one staging buffer each, in
    // one submit, then wait once.
    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("hanabi:bookkeeping_probe"),
    });
    // Every child's event count, in one copy of the whole array.
    let child_infos = event_cache.child_infos_buffer().map(|buffer| {
        let staging = render_device.create_buffer(&BufferDescriptor {
            label: Some("hanabi:bookkeeping_probe:child_infos"),
            size: buffer.size(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
        staging
    });
    let mut readbacks = Vec::new();
    for (main_entity, cached_effect, cached_metadata, child_info) in &q_effects {
        let Some(slab) = effect_cache.get_slab(&cached_effect.slab_id) else {
            continue;
        };
        let range = cached_effect.slice.range();
        let rows_size = (range.end - range.start) as u64 * row_size;
        let layout = &cached_effect.slice.particle_layout;
        let stride = layout.min_binding_size().get();
        let particles_size = (range.end - range.start) as u64 * stride;
        let age_offset = layout.byte_offset(Attribute::AGE);
        let staging = render_device.create_buffer(&BufferDescriptor {
            label: Some("hanabi:bookkeeping_probe"),
            size: metadata_size + rows_size + particles_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let metadata_offset = effects_meta
            .effect_metadata_buffer
            .dynamic_offset(cached_metadata.table_id) as u64;
        encoder.copy_buffer_to_buffer(metadata_buffer, metadata_offset, &staging, 0, metadata_size);
        encoder.copy_buffer_to_buffer(
            slab.indirect_index_buffer(),
            range.start as u64 * row_size,
            &staging,
            metadata_size,
            rows_size,
        );
        encoder.copy_buffer_to_buffer(
            slab.particle_buffer(),
            range.start as u64 * stride,
            &staging,
            metadata_size + rows_size,
            particles_size,
        );
        let ages = age_offset.map(|offset| (stride as usize, offset as usize));
        readbacks.push((main_entity.id(), range, ages, child_info.is_some(), staging));
    }
    if readbacks.is_empty() {
        return;
    }
    render_queue.submit([encoder.finish()]);
    for (_, _, _, _, staging) in &readbacks {
        staging
            .slice(..)
            .map_async(MapMode::Read, |result| result.unwrap());
    }
    if let Some(staging) = &child_infos {
        staging
            .slice(..)
            .map_async(MapMode::Read, |result| result.unwrap());
    }
    if let Err(err) = render_device.poll(PollType::wait_indefinitely()) {
        warn!("Bookkeeping probe: failed to wait for the GPU: {err:?}");
        return;
    }

    let child_infos: Option<Vec<GpuChildInfo>> = child_infos.as_ref().and_then(|staging| {
        let data = staging.slice(..).get_mapped_range().ok()?;
        Some(bytemuck::cast_slice(&data[..]).to_vec())
    });
    let mut samples = samples.0.lock().unwrap();
    for (entity, range, ages, is_child, staging) in &readbacks {
        let Ok(data) = staging.slice(..).get_mapped_range() else {
            continue;
        };
        let metadata: GpuEffectMetadata =
            bytemuck::pod_read_unaligned(&data[..metadata_size as usize]);
        // Nothing has spawned yet: the metadata may not even be uploaded.
        if metadata.particle_counter == 0 {
            continue;
        }
        let rows_end = metadata_size as usize + (range.end - range.start) as usize * 12;
        let rows: &[[u32; 3]] = bytemuck::cast_slice(&data[metadata_size as usize..rows_end]);
        let mut sample = check_lists(clock.frame, *entity, range.start, &metadata, rows);
        if *is_child {
            sample.events = child_infos
                .as_ref()
                .and_then(|infos| infos.get(metadata.global_child_index as usize))
                .map(|info| info.event_count);
        }
        // The GPU's stride must match the layout's for the copy to line up.
        let ages = ages.filter(|(stride, _)| *stride == metadata.particle_stride as usize * 4);
        if let Some((stride, offset)) = ages {
            let particles = &data[rows_end..];
            let column = metadata.indirect_write_index as usize & 1;
            let alive = metadata.alive_count.min(rows.len() as u32) as usize;
            sample.oldest_age = rows[..alive]
                .iter()
                .filter_map(|row| {
                    let at = row[column] as usize * stride + offset;
                    particles
                        .get(at..at + 4)
                        .map(bytemuck::pod_read_unaligned::<f32>)
                })
                .reduce(f32::max);
        }
        if sample.duplicated > 0 || sample.leaked > 0 {
            warn!("Bookkeeping probe: slots lost or listed twice: {sample:?}");
        } else {
            debug!("Bookkeeping probe: {sample:?}");
        }
        samples.push(sample);
    }
}

/// Count how many times each slot of one effect is listed. `base_particle` is
/// the effect's first row in its slab: alive entries are relative to it, dead
/// entries are slab indices.
fn check_lists(
    frame: u32,
    entity: Entity,
    base_particle: u32,
    metadata: &GpuEffectMetadata,
    rows: &[[u32; 3]],
) -> BookkeepingSample {
    let capacity = rows.len() as u32;
    // The update pass wrote the alive list into the column named by
    // indirect_write_index, after the indirect pass swapped it this frame.
    let column = metadata.indirect_write_index as usize & 1;
    let alive_count = metadata.alive_count.min(capacity) as usize;
    let mut listed = vec![0u32; rows.len()];
    for row in &rows[..alive_count] {
        if let Some(n) = listed.get_mut(row[column] as usize) {
            *n += 1;
        }
    }
    for row in &rows[alive_count..] {
        if let Some(n) = row[2]
            .checked_sub(base_particle)
            .and_then(|index| listed.get_mut(index as usize))
        {
            *n += 1;
        }
    }
    BookkeepingSample {
        frame,
        entity,
        capacity,
        alive_count: metadata.alive_count,
        duplicated: listed.iter().filter(|&&n| n > 1).count() as u32,
        leaked: listed.iter().filter(|&&n| n == 0).count() as u32,
        oldest_age: None,
        events: None,
    }
}
