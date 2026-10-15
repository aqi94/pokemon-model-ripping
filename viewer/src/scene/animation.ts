import * as THREE from 'three';
import type { ClipInfo } from '../types';

export interface PlaybackState {
  clip: string | null;
  time: number;
  duration: number;
  frame: number;
  frameCount: number;
  paused: boolean;
  timeScale: number;
}

export const TIME_SCALES = [0.25, 0.5, 1, 2];

export class AnimationController {
  mixer: THREE.AnimationMixer;
  private clips = new Map<string, THREE.AnimationClip>();
  private infos = new Map<string, ClipInfo>();
  private action: THREE.AnimationAction | null = null;
  current: string | null = null;
  paused = false;
  timeScale = 1;

  constructor(root: THREE.Object3D, clips: THREE.AnimationClip[], infos: ClipInfo[]) {
    this.mixer = new THREE.AnimationMixer(root);
    clips.forEach((c) => this.clips.set(c.name, c));
    infos.forEach((i) => this.infos.set(i.name, i));
  }

  has(name: string) { return this.clips.has(name); }

  play(name: string) {
    const clip = this.clips.get(name);
    if (!clip) return;
    const info = this.infos.get(name);
    const prev = this.action;
    const next = this.mixer.clipAction(clip);
    next.reset();
    next.enabled = true;
    if (info?.loop === false) {
      next.setLoop(THREE.LoopOnce, 1);
      next.clampWhenFinished = true;
    } else {
      next.setLoop(THREE.LoopRepeat, Infinity);
      next.clampWhenFinished = false;
    }
    if (prev && prev !== next) prev.stop();
    next.paused = this.paused;
    next.play();
    this.action = next;
    this.current = name;
    this.mixer.update(0);
  }

  update(dt: number) {
    this.mixer.timeScale = this.timeScale;
    this.mixer.update(dt);
  }

  setPaused(p: boolean) {
    this.paused = p;
    const a = this.action;
    if (!a) return;
    if (!p && a.loop === THREE.LoopOnce && a.time >= a.getClip().duration - 1e-4) {
      a.reset(); // replay a finished one-shot
      a.play();
    }
    a.paused = p;
  }

  setTimeScale(s: number) { this.timeScale = s; }

  scrubToFrame(frame: number) {
    const a = this.action;
    if (!a) return;
    const info = this.infos.get(this.current!);
    const dur = a.getClip().duration;
    const fc = Math.max(1, (info?.frameCount ?? 1) - 1);
    a.enabled = true;
    a.paused = true;
    a.time = Math.min(dur, (frame / fc) * dur);
    this.paused = true;
    const ts = this.mixer.timeScale;
    this.mixer.update(0);
    this.mixer.timeScale = ts;
  }

  state(): PlaybackState {
    const a = this.action;
    const info = this.current ? this.infos.get(this.current) : undefined;
    const dur = a?.getClip().duration ?? 0;
    const time = a ? Math.min(dur, Math.max(0, a.time)) : 0;
    const fcount = info?.frameCount ?? 0;
    const frame = dur > 0 ? Math.round((time / dur) * Math.max(0, fcount - 1)) : 0;
    return { clip: this.current, time, duration: dur, frame, frameCount: fcount, paused: this.paused, timeScale: this.timeScale };
  }
}
