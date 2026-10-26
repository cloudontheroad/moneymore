import assert from "node:assert/strict";
import test from "node:test";
import {latestSnapshot,isSnapshotStale,captureHelp,jobLabels} from "../app/desktop-view.ts";

const now=Date.parse("2026-10-26T11:16:00+08:00");
const snapshot=(overrides={})=>({id:"new",kind:"positions",account_ref:"test",data_date:"20261026",source_name:"positions.xls",source_mtime:"2026-10-26T11:15:00+08:00",imported_at:"2026-10-26T11:15:01+08:00",row_count:8,freshness:"RECENT_UNVERIFIED",rows:[],rows_truncated:false,...overrides});
test("latest data is chosen by source time, not import time or array order",()=>{
  const fresh=snapshot();const old=snapshot({id:"old",source_mtime:"2026-10-22T11:15:00+08:00",imported_at:"2026-10-26T11:17:00+08:00"});
  const input=[old,fresh];assert.equal(latestSnapshot(input,"positions").id,"new");assert.equal(input[0].id,"old");assert.equal(latestSnapshot(input,"trades"),undefined);
  assert.equal(latestSnapshot([snapshot({source_mtime:"invalid"}),fresh],"positions").id,"new");
});
test("stale, wrong-date, future or invalid snapshots never look fresh",()=>{
  assert.equal(isSnapshotStale(snapshot(),now),false);
  for(const s of [undefined,snapshot({freshness:"STALE"}),snapshot({data_date:"20261025"}),snapshot({source_mtime:"invalid"}),snapshot({source_mtime:"2026-10-26T12:00:00+08:00"})])assert.equal(isSnapshotStale(s,now),true);
  assert.equal(isSnapshotStale(snapshot(),now+300001),true);
});
test("zero rows stay a genuine zero while missing snapshots remain absent",()=>{
  assert.equal(latestSnapshot([snapshot({kind:"orders",row_count:0})],"orders").row_count,0);
  assert.equal(latestSnapshot([],"orders"),undefined);
});
test("success and failures retain explicit read-only and verification boundaries",()=>{
  assert.match(jobLabels.COMPLETED_UNVERIFIED,/身份待核验/);
  assert.match(captureHelp("FOREGROUND_CHANGED"),/前台/);
  assert.match(captureHelp("UNKNOWN_CODE"),/安全停止/);
});
