"use strict";

const assert = require("node:assert/strict");
const { LESSONS } = require("./notebook.js");

assert.deepEqual(LESSONS.blind.moved, [0, 1, 2, 3, 4, 5, 6, 7]);
assert.equal(LESSONS.blind.applications / LESSONS.blind.moved.length, 0.25);
assert.deepEqual(LESSONS.route.moved, [2, 6]);
assert.equal(LESSONS.route.applications / LESSONS.route.moved.length, 1);
assert.deepEqual(LESSONS.reuse.moved, [2, 5, 6]);
assert.equal(LESSONS.reuse.applications, 4);

console.log("notebook lesson model: ok");
