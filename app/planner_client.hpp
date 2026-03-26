#pragma once

#include <memory>
#include <string>

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TTransportUtils.h>

#include "gen-cpp/CalciteServer.h"
#include "gen-cpp/calciteserver_types.h"

namespace app
{

class PlannerClient
{
private:
    std::shared_ptr<apache::thrift::transport::TTransport> socket_;
    std::shared_ptr<apache::thrift::transport::TTransport> transport_;
    std::shared_ptr<apache::thrift::protocol::TProtocol> protocol_;
    CalciteServerClient client_;
public:
    PlannerClient(const std::string &host, int port)
        : socket_(std::make_shared<apache::thrift::transport::TSocket>(host, port)),
          transport_(std::make_shared<apache::thrift::transport::TBufferedTransport>(socket_)),
          protocol_(std::make_shared<apache::thrift::protocol::TBinaryProtocol>(transport_)),
          client_(protocol_)
    {
    }

    void open()
    {
        transport_->open();
    }

    void close()
    {
        if (transport_->isOpen())
            transport_->close();
    }

    void ping()
    {
        client_.ping();
    }

    void parse(PlanResult &result, const std::string &sql)
    {
        client_.parse(result, sql);
    }

    ~PlannerClient()
    {
        try
        {
            close();
        }
        catch (...)
        {
        }
    }
};

} // namespace app
