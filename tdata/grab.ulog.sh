#!/bin/sh

PCAP=/var/log/ulog/ulogd.pcap

if test $# -ge "2"; then
  PCAP=$2;
fi;

echo "using $PCAP";

tshark -F pcap -Y 'tcp and tcp.dstport eq 12345' -r $PCAP  -w - | editcap -F pcap -i600 - tmp/"$1".pcap
