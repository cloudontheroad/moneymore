export type ExportKind = "positions" | "orders" | "trades";
export type BrokerSnapshot = {
  id:string;kind:ExportKind;account_ref:string;data_date:string;source_name:string;
  source_mtime:string;imported_at:string;row_count:number;freshness:string;
  rows_truncated:boolean;rows:Record<string,string|number|null>[];
};
export const kindLabels:Record<ExportKind,string>={positions:"持仓",orders:"委托",trades:"成交"};
export const jobLabels:Record<string,string>={QUEUED:"等待开始",RUNNING:"同步中",STOP_REQUESTED:"正在停止",STOPPED:"已停止",INTERRUPTED:"任务已中断",BLOCKED:"同步未完成",PARTIAL_IMPORT:"部分数据已保存",COMPLETED_UNVERIFIED:"同步完成 · 身份待核验"};
export function latestSnapshot(snapshots:BrokerSnapshot[],kind:ExportKind){
  return snapshots.filter(s=>s.kind===kind&&Number.isFinite(Date.parse(s.source_mtime))).sort((a,b)=>Date.parse(b.source_mtime)-Date.parse(a.source_mtime))[0];
}
export function isSnapshotStale(snapshot:BrokerSnapshot|undefined,now:number){
  if(!snapshot)return true;
  const today=new Date(now).toLocaleDateString("sv-SE",{timeZone:"Asia/Shanghai"}).replaceAll("-","");
  const stamp=Date.parse(snapshot.source_mtime);
  return snapshot.freshness==="STALE"||snapshot.data_date!==today||!Number.isFinite(stamp)||stamp>now||now-stamp>300000;
}
export function dateTime(value:string|undefined){
  if(!value||!Number.isFinite(Date.parse(value)))return "尚无记录";
  return new Date(value).toLocaleString("zh-CN",{timeZone:"Asia/Shanghai",month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",second:"2-digit",hour12:false});
}
export function captureHelp(reason?:string){
  const map:Record<string,string>={FOREGROUND_CHANGED:"前台被切换。重新同步期间请勿操作键鼠。",USER_INPUT_ACTIVE:"检测到键鼠操作。空闲时再发起同步。",DESKTOP_LOCKED_OR_UNAVAILABLE:"桌面锁定或不可访问，请手动解锁。",UNEXPECTED_DIALOG:"出现未识别窗口，请手动处理华泰弹窗。",LEASE_EXPIRED_OR_OWNER_EXITED:"服务或授权已中断，旧任务不会自动续跑。",SELECTED_ACCOUNT_CHANGED:"所选账户发生变化，确认账户后重新同步。",WORKER_TIMEOUT:"同步超过时限，请检查客户端是否响应。",SAVE_DIALOG_DID_NOT_CLOSE:"保存窗口未正常关闭，请检查华泰提示。",CLIENT_PROCESS_NOT_UNIQUE:"未找到唯一客户端，请手动登录华泰并避免多开。",STOP_REQUESTED:"你已停止本次同步，未发出交易。"};
  return reason?(map[reason]??"同步遇到不兼容状态，已安全停止。高级区可查看诊断代码。"):"请确认华泰已登录、桌面已解锁，再尝试同步。";
}
