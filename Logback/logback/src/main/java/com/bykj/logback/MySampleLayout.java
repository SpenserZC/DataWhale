package com.bykj.logback;

import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.LayoutBase;
import net.sf.json.JSONObject;

public class MySampleLayout extends LayoutBase<ILoggingEvent> {

  public String doLayout(ILoggingEvent event) {
    StringBuffer sbuf = new StringBuffer(128);
    sbuf.append("{Timestamp:");
    sbuf.append(event.getTimeStamp()+",");
    sbuf.append("ThreadName:");
    sbuf.append(event.getThreadName()+",");
    sbuf.append("fullLink:");
    sbuf.append(JSONObject.fromObject(event.getMDCPropertyMap()));
    return sbuf.toString();
  }  
  
}
