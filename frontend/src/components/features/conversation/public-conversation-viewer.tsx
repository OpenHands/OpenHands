import React from "react";
import { useTranslation } from "react-i18next";
import { Card, CardContent, CardHeader, CardTitle } from "#/components/ui/card";
import { Badge } from "#/components/ui/badge";
import { Separator } from "#/components/ui/separator";
import { Alert, AlertDescription } from "#/components/ui/alert";
import { Globe, Calendar, GitBranch, User, Bot, AlertCircle } from "lucide-react";
import { usePublicConversationFull } from "#/hooks/query/use-public-conversation-sharing";
import { formatDistanceToNow } from "date-fns";

interface PublicConversationViewerProps {
  conversationId: string;
}

export function PublicConversationViewer({ conversationId }: PublicConversationViewerProps) {
  const { t } = useTranslation();
  const { data: conversationDetail, isLoading, error } = usePublicConversationFull(conversationId);

  if (isLoading) {
    return (
      <div className="max-w-4xl mx-auto p-6 space-y-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-3/4 mb-4"></div>
          <div className="h-4 bg-gray-200 rounded w-1/2 mb-6"></div>
          <div className="space-y-4">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="h-24 bg-gray-200 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error || !conversationDetail) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <Alert className="border-red-200 bg-red-50">
          <AlertCircle className="h-4 w-4 text-red-600" />
          <AlertDescription className="text-red-800">
            {t("PUBLIC_CONVERSATION.NOT_FOUND")}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  const { conversation, messages } = conversationDetail;

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="text-center space-y-4">
        <div className="flex items-center justify-center gap-2 text-green-600">
          <Globe className="h-5 w-5" />
          <span className="text-sm font-medium">{t("PUBLIC_CONVERSATION.PUBLIC_BADGE")}</span>
        </div>
        <h1 className="text-3xl font-bold text-gray-900">
          {conversation.title}
        </h1>
        <p className="text-gray-600">
          {t("PUBLIC_CONVERSATION.SHARED_CONVERSATION")}
        </p>
      </div>

      {/* Conversation Metadata */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">
            {t("PUBLIC_CONVERSATION.CONVERSATION_INFO")}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex items-center gap-2">
              <Calendar className="h-4 w-4 text-gray-500" />
              <span className="text-sm">
                <strong>{t("PUBLIC_CONVERSATION.CREATED")}:</strong>{" "}
                {formatDistanceToNow(new Date(conversation.created_at), { addSuffix: true })}
              </span>
            </div>

            {conversation.selected_repository && (
              <div className="flex items-center gap-2">
                <GitBranch className="h-4 w-4 text-gray-500" />
                <span className="text-sm">
                  <strong>{t("PUBLIC_CONVERSATION.REPOSITORY")}:</strong>{" "}
                  {conversation.selected_repository}
                  {conversation.selected_branch && ` (${conversation.selected_branch})`}
                </span>
              </div>
            )}

            {conversation.trigger && (
              <div className="flex items-center gap-2">
                <Badge variant="secondary" className="text-xs">
                  {conversation.trigger}
                </Badge>
              </div>
            )}

            <div className="flex items-center gap-2">
              <Badge
                variant={conversation.status === "RUNNING" ? "default" : "secondary"}
                className="text-xs"
              >
                {conversation.status}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Messages */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">
            {t("PUBLIC_CONVERSATION.CONVERSATION_MESSAGES")} ({messages.length})
          </CardTitle>
        </CardHeader>
        <CardContent>
          {messages.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              {t("PUBLIC_CONVERSATION.NO_MESSAGES")}
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((message, index) => (
                <div key={message.id} className="space-y-2">
                  <div className="flex items-center gap-2">
                    {message.source === "user" ? (
                      <User className="h-4 w-4 text-blue-600" />
                    ) : (
                      <Bot className="h-4 w-4 text-green-600" />
                    )}
                    <span className="text-sm font-medium capitalize">
                      {message.source === "user" ? t("PUBLIC_CONVERSATION.USER") : t("PUBLIC_CONVERSATION.ASSISTANT")}
                    </span>
                    <span className="text-xs text-gray-500">
                      {formatDistanceToNow(new Date(message.timestamp), { addSuffix: true })}
                    </span>
                  </div>
                  <div className="ml-6 p-3 bg-gray-50 rounded-lg">
                    <pre className="whitespace-pre-wrap text-sm text-gray-800 font-sans">
                      {message.content}
                    </pre>
                  </div>
                  {index < messages.length - 1 && <Separator className="my-4" />}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Footer */}
      <div className="text-center text-sm text-gray-500 space-y-2">
        <p>{t("PUBLIC_CONVERSATION.POWERED_BY")} OpenHands</p>
        <p>{t("PUBLIC_CONVERSATION.READ_ONLY_NOTICE")}</p>
      </div>
    </div>
  );
}
