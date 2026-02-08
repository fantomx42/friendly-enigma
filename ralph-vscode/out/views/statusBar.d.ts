import { RalphRunner } from "../ralph/ralphRunner";
/**
 * Status bar item that shows Ralph's current state.
 * Click opens the output channel.
 */
export declare class StatusBar {
    private runner;
    private item;
    constructor(runner: RalphRunner);
    private update;
    dispose(): void;
}
